import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatDuration,
  validateAudioFile,
  MAX_AUDIO_BYTES,
} from "@/lib/audioValidation";

describe("validateAudioFile", () => {
  function fakeFile(name: string, type: string, size: number): File {
    const blob = new Blob([new ArrayBuffer(size)], { type });
    return new File([blob], name, { type });
  }

  it("accepts WAV by MIME", () => {
    const f = fakeFile("a.wav", "audio/wav", 1024);
    expect(validateAudioFile(f).ok).toBe(true);
  });

  it("accepts WAV by extension when MIME is empty", () => {
    const f = fakeFile("a.wav", "", 1024);
    expect(validateAudioFile(f).ok).toBe(true);
  });

  it("rejects non-WAV MIME", () => {
    const f = fakeFile("a.mp3", "audio/mpeg", 1024);
    const v = validateAudioFile(f);
    expect(v.ok).toBe(false);
    expect(v.reason).toMatch(/format/i);
  });

  it("rejects oversized files", () => {
    const f = fakeFile("a.wav", "audio/wav", MAX_AUDIO_BYTES + 1);
    const v = validateAudioFile(f);
    expect(v.ok).toBe(false);
    expect(v.reason).toMatch(/za duży/i);
  });
});

describe("formatBytes", () => {
  it("formats KB and MB", () => {
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.00 MB");
  });
});

describe("formatDuration", () => {
  it("formats mm:ss with zero padding", () => {
    expect(formatDuration(0)).toBe("00:00");
    expect(formatDuration(9.5)).toBe("00:09");
    expect(formatDuration(65)).toBe("01:05");
    expect(formatDuration(3661)).toBe("61:01");
  });
});
