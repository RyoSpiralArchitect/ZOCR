from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from ..service.executor import execute_command, watch_parent


_SWIFT_SOURCE = r'''
import Foundation
import Vision
import ImageIO
import Darwin
import Dispatch

let parent = Int32(CommandLine.arguments[1])!
DispatchQueue.global().async {
    while getppid() == parent { usleep(100000) }
    kill(-getpgrp(), SIGKILL)
}
do {
    let input = try JSONSerialization.jsonObject(with: FileHandle.standardInput.readDataToEndOfFile()) as! [String: Any]
    let bytes = Data(base64Encoded: input["png"] as! String)!
    let source = CGImageSourceCreateWithData(bytes as CFData, nil)!
    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let languages = input["languages"] as! [String]
    let supported = try request.supportedRecognitionLanguages()
    guard !languages.isEmpty && languages.allSatisfy({ supported.contains($0) }) else {
        throw NSError(domain: "Unsupported OCR language", code: 1)
    }
    request.recognitionLanguages = languages
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let observations: [[String: Any]] = (request.results ?? []).compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return ["text": candidate.string, "confidence": candidate.confidence,
                "bbox": [box.minX, box.minY, box.width, box.height]]
    }
    let output: [String: Any] = ["engine": "apple_vision", "revision": request.revision,
        "languages": languages, "observations": observations,
        "os_version": ProcessInfo.processInfo.operatingSystemVersionString]
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]))
} catch {
    FileHandle.standardError.write(Data("Vision OCR failed: \(error)\n".utf8))
    exit(1)
}
'''


def validate_response(response: dict, width: int, height: int) -> dict:
    if response.get("engine") != "apple_vision" or response.get("revision") != 3:
        raise ValueError("Unexpected OCR provider identity")
    observations = response.get("observations")
    if not isinstance(observations, list) or len(observations) > 10000:
        raise ValueError("Invalid OCR region count")
    lines = []
    for observation in observations:
        text = observation.get("text")
        confidence = observation.get("confidence")
        box = observation.get("bbox")
        if (not isinstance(text, str) or len(text) > 100000 or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or not isinstance(box, list) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in box)):
            raise ValueError("Invalid OCR observation")
        x, y, w, h = box
        if (x < -1 / width or y < -1 / height or min(w, h) <= 0
                or x + w > 1 + 1 / width or y + h > 1 + 1 / height):
            raise ValueError("OCR coordinates exceed the source image")
        pixels = [max(0, math.floor(x * width)), max(0, math.floor((1 - y - h) * height)),
                  min(width, math.ceil((x + w) * width)), min(height, math.ceil((1 - y) * height))]
        if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
            raise ValueError("Empty OCR region")
        lines.append({"text": text, "confidence": float(confidence), "bbox": pixels,
                      "normalized_bbox": box, "bbox_clipped": x < 0 or y < 0 or x + w > 1 or y + h > 1})
    return {**response, "lines": lines, "provider_source_sha256": hashlib.sha256(_SWIFT_SOURCE.encode()).hexdigest(),
            "model_identity": "OS-managed Apple Vision revision 3; model weight hash unavailable",
            "model_sha256": None, "language_correction": False, "device_policy": "system_default"}


class VisionRecognizer:
    def __init__(self, *, languages=("ja-JP", "en-US"), timeout_sec: float = 30):
        if platform.system() != "Darwin" or shutil.which("swiftc") is None:
            raise RuntimeError("Apple Vision requires macOS and the Swift compiler; no OCR fallback was selected")
        if (not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 300
                or not languages or any(not isinstance(value, str) or not value for value in languages)):
            raise ValueError("Invalid Vision OCR configuration")
        self.languages = list(languages)
        self.timeout_sec = timeout_sec
        self._temporary = tempfile.TemporaryDirectory(prefix="zocr-vision-")
        self.binary = Path(self._temporary.name) / "vision-ocr"
        try:
            self.build = execute_command([sys.executable, "-m", __name__, "--compile", str(os.getpid())],
                                         {"directory": self._temporary.name}, 120)
        except BaseException:
            self.close()
            raise

    def recognize(self, image) -> dict:
        if image.width * image.height > 40_000_000:
            raise ValueError("OCR image exceeds the pixel budget")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("OCR image exceeds the byte budget")
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != self.build["binary_sha256"]:
            raise ValueError("Vision bridge changed after compilation")
        response = execute_command([str(self.binary), str(os.getpid())],
                                   {"png": base64.b64encode(raw).decode("ascii"), "languages": self.languages}, self.timeout_sec)
        if response.get("languages") != self.languages:
            raise ValueError("OCR provider did not use the requested languages")
        return {**validate_response(response, image.width, image.height), "build": self.build}

    def close(self):
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _compile(directory: str) -> dict:
    root = Path(directory)
    source, binary = root / "vision.swift", root / "vision-ocr"
    source.write_text(_SWIFT_SOURCE, encoding="utf-8")
    subprocess.run(["swiftc", "-module-cache-path", str(root / "modules"), str(source), "-o", str(binary)],
                   check=True, stdout=sys.stderr)
    return {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(_SWIFT_SOURCE.encode()).hexdigest(),
            "platform": platform.platform()}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--compile":
        raise SystemExit("Use the VisionRecognizer interface")
    watch_parent(int(sys.argv[2]))
    payload = json.load(sys.stdin)
    result = _compile(payload["directory"])
    sys.stdout.write(json.dumps(result, allow_nan=False))
