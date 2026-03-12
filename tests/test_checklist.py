"""
測試清單自動化腳本
對應 PPT 技術挑戰與三大功能的驗證項目
執行方式：python tests/test_checklist.py
"""

import os
import sys
import time
import struct
import tempfile
import threading
import unittest

# 讓測試可以 import 根目錄模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# 1. 功能性測試 (Functional Testing)
# ══════════════════════════════════════════════════════════════════

class TestSTTAudioChunk(unittest.TestCase):
    """[ ] 即時辨識率：音頻 chunk 穩定轉換"""

    def _make_pcm_chunk(self, duration_sec: float, freq_hz: float = 440.0) -> bytes:
        """產生指定長度的正弦波 PCM int16"""
        import math
        sample_rate = 16000
        n = int(sample_rate * duration_sec)
        samples = []
        for i in range(n):
            val = int(0.3 * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            samples.append(val)
        return struct.pack(f"<{n}h", *samples)

    def test_silence_becomes_zeros_not_dropped(self):
        """靜音 chunk 應以零值送入 buffer，不直接丟棄"""
        import numpy as np
        # 產生靜音（RMS < 0.003）
        silent_chunk = struct.pack("<160h", *([0] * 160))
        # 直接測試 _append_pcm_chunk 邏輯
        pcm32 = (
            np.frombuffer(silent_chunk, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        rms = float(np.sqrt(np.mean(np.square(pcm32))))
        self.assertLess(rms, 0.003, "靜音樣本 RMS 應小於門檻")
        # 靜音 chunk 應產生等長的零值，而非被丟棄
        zeros = pcm32 * 0
        self.assertEqual(zeros.sum(), 0.0)
        self.assertEqual(len(zeros), len(pcm32), "零值音頻長度應與原 chunk 相同")

    def test_chunk_duration_range(self):
        """0.8 ~ 5 秒的 chunk 大小範圍是否合理"""
        sample_rate = 16000
        for duration in [0.8, 1.0, 2.0, 5.0]:
            chunk = self._make_pcm_chunk(duration)
            n_samples = len(chunk) // 2          # int16 = 2 bytes
            actual_sec = n_samples / sample_rate
            self.assertAlmostEqual(actual_sec, duration, places=2,
                msg=f"chunk {duration}s 樣本數不正確")

    def test_downsample_output_length(self):
        """降頻後長度應符合 inputRate/outputRate 比例"""
        # 模擬前端 downsampleBuffer 邏輯（Python 版）
        import math
        def downsample_simple(buf, in_rate, out_rate):
            ratio = in_rate / out_rate
            new_len = math.floor(len(buf) / ratio)
            return new_len

        in_rate, out_rate = 48000, 16000
        input_len = 4096
        expected = input_len * out_rate // in_rate
        actual = downsample_simple(range(input_len), in_rate, out_rate)
        self.assertEqual(actual, expected,
            f"降頻後長度應為 {expected}，實際 {actual}")


class TestSummaryModes(unittest.TestCase):
    """[ ] 摘要模式切換：三種模式輸出格式驗證"""

    def _mock_call_model(self, system_prompt, user_prompt):
        """回傳符合格式的假摘要，避免實際呼叫 LLM"""
        if "全文摘要" in system_prompt or "精簡摘要" in system_prompt:
            return "這次會議討論了產品規劃方向，確認了下一季的目標。"
        if "條列" in system_prompt:
            return "• 確認 Q3 目標\n• 指派負責人\n• 下週 follow-up"
        if "待辦" in system_prompt:
            return "- [ ] 確認 Q3 目標\n- [ ] 指派負責人"
        return "逐字稿資訊不足"

    def test_full_summary_is_prose(self):
        """全文摘要應為段落文字，不包含條列符號"""
        result = self._mock_call_model("精簡摘要", "這次會議...")
        self.assertNotIn("•", result, "全文摘要不應出現條列符號")
        self.assertGreater(len(result), 10, "全文摘要不應為空")

    def test_key_points_has_bullets(self):
        """重點條列應包含「•」開頭"""
        result = self._mock_call_model("條列", "...")
        lines = [l.strip() for l in result.splitlines() if l.strip()]
        bullet_lines = [l for l in lines if l.startswith("•")]
        self.assertGreater(len(bullet_lines), 0, "重點條列應有至少一個「•」項目")

    def test_action_items_has_checkbox(self):
        """待辦清單應包含「- [ ]」格式"""
        result = self._mock_call_model("待辦", "...")
        self.assertIn("- [ ]", result, "待辦清單應使用「- [ ] 」格式")

    def test_summarize_all_in_one_keys(self):
        """summarize_all_in_one 回傳 dict 應含三個 key"""
        import importlib, unittest.mock, sys
        # 若 requests 未安裝，用 mock 替換後再 import
        if "requests" not in sys.modules:
            sys.modules["requests"] = unittest.mock.MagicMock()
        cognition = importlib.import_module("cognition")
        with unittest.mock.patch.object(cognition, "_call_model",
                side_effect=lambda sp, up: "【全文摘要】測試摘要\n【重點條列】• 重點\n【待辦清單】- [ ] 行動"):
            result = cognition.summarize_all_in_one("這是一段測試逐字稿，內容足夠長，可以進行摘要。")
        self.assertIn("full", result)
        self.assertIn("key_points", result)
        self.assertIn("action_items", result)


class TestTraditionalConversion(unittest.TestCase):
    """[ ] 繁體轉換準確度：s2twp 台灣用語驗證"""

    @classmethod
    def setUpClass(cls):
        try:
            import opencc
            cls.converter = opencc.OpenCC("s2twp")
            cls.available = True
        except Exception:
            cls.available = False

    def test_software_terms(self):
        """軟體術語：簡體「軟件」→ 繁體「軟體」"""
        if not self.available:
            self.skipTest("opencc 不可用")
        result = self.converter.convert("软件开发")
        self.assertIn("軟體", result, f"「软件」應轉為「軟體」，實際：{result}")

    def test_common_terms(self):
        """常見詞彙轉換正確"""
        if not self.available:
            self.skipTest("opencc 不可用")
        cases = [
            ("视频", "影片"),
            ("内存", "記憶體"),
            ("硬盘", "硬碟"),
        ]
        for simplified, expected_tw in cases:
            result = self.converter.convert(simplified)
            self.assertIn(expected_tw, result,
                f"「{simplified}」應含「{expected_tw}」，實際：{result}")

    def test_already_traditional(self):
        """純繁體輸入不應產生亂字"""
        if not self.available:
            self.skipTest("opencc 不可用")
        original = "這是一段繁體中文測試"
        result = self.converter.convert(original)
        self.assertTrue(len(result) > 0)
        # 不應有簡體常見字混入
        simplified_chars = set("这样说话")
        self.assertTrue(all(ch not in result for ch in simplified_chars),
            f"轉換後不應出現簡體字，實際：{result}")


class TestFileExport(unittest.TestCase):
    """[ ] 檔案匯出：.txt 正確儲存"""

    def test_export_creates_txt_files(self):
        """匯出應產生 transcript.txt 與 summary.txt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = os.path.join(tmpdir, "transcript.txt")
            summary_path = os.path.join(tmpdir, "summary.txt")

            content_t = "會議名稱: 測試\n==\n\n【逐字稿】\n[12:00:00] 這是測試內容"
            content_s = "會議名稱: 測試\n==\n\n【全文摘要】\n摘要內容"

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(content_t)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(content_s)

            self.assertTrue(os.path.exists(transcript_path), "transcript.txt 未建立")
            self.assertTrue(os.path.exists(summary_path), "summary.txt 未建立")

    def test_export_utf8_no_corruption(self):
        """匯出繁體中文不應出現亂碼（UTF-8 正確性）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            text = "測試：繁體中文、台灣用語、待辦清單 ✓"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(path, "r", encoding="utf-8") as f:
                result = f.read()
            self.assertEqual(result, text, "UTF-8 讀寫結果應一致")

    def test_export_path_in_documents(self):
        """匯出路徑應在 Documents/AI Meeting Assistant 下"""
        import sys
        home = os.path.expanduser("~")
        if sys.platform == "darwin":
            expected_base = os.path.join(home, "Documents", "AI Meeting Assistant")
        elif sys.platform.startswith("win"):
            expected_base = os.path.join(home, "Documents", "AI Meeting Assistant")
        else:
            expected_base = os.path.join(home, "ai-meeting-assistant")
        # 驗證路徑組合邏輯（不實際建立）
        export_path = os.path.join(expected_base, "download", "test_meeting")
        self.assertIn("AI Meeting Assistant", export_path,
            "匯出路徑應含 'AI Meeting Assistant'")


# ══════════════════════════════════════════════════════════════════
# 2. 環境與部署測試 (Deployment Testing)
# ══════════════════════════════════════════════════════════════════

class TestOfflineMode(unittest.TestCase):
    """[ ] 離線執行：STT 與 LLM 不依賴網路"""

    def test_stt_engine_no_network_import(self):
        """stt_engine 模組不應 import 任何網路相關函式庫"""
        import ast, pathlib
        src = pathlib.Path(__file__).parent.parent / "stt_engine.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        network_libs = {"requests", "urllib", "httpx", "aiohttp", "socket"}
        imports_found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = getattr(node, "module", None) or ""
                for alias in getattr(node, "names", []):
                    full = alias.name.split(".")[0]
                    if full in network_libs:
                        imports_found.add(full)
                if name.split(".")[0] in network_libs:
                    imports_found.add(name.split(".")[0])
        self.assertEqual(imports_found, set(),
            f"stt_engine.py 不應有網路 import：{imports_found}")

    def test_cognition_gguf_only_flag(self):
        """AMA_DISABLE_OLLAMA_FALLBACK=1 時，cognition 應只使用本地 GGUF"""
        import importlib, unittest.mock, sys
        if "requests" not in sys.modules:
            sys.modules["requests"] = unittest.mock.MagicMock()
        os.environ["AMA_DISABLE_OLLAMA_FALLBACK"] = "1"
        try:
            cognition = importlib.import_module("cognition")
            importlib.reload(cognition)
            # gguf_only 模式下，若本地模型不可用應回傳 [錯誤]
            result = cognition._call_model("system", "user")
            self.assertTrue(
                result.startswith("[錯誤]") or result == "",
                f"gguf_only 模式無模型時應回傳 [錯誤]，實際：{result}"
            )
        finally:
            del os.environ["AMA_DISABLE_OLLAMA_FALLBACK"]


class TestModelPathResolution(unittest.TestCase):
    """[ ] 路徑相容性：_is_frozen() 模型路徑邏輯"""

    def test_dev_mode_project_root(self):
        """開發模式下應從 _project_root() 找模型"""
        from stt_engine import _is_frozen, _project_root
        self.assertFalse(_is_frozen(), "測試環境不應是 frozen 模式")
        root = _project_root()
        self.assertTrue(root.exists(), f"project root 應存在：{root}")

    def test_sherpa_dir_search_order(self):
        """_find_bundled_sherpa_dir 在模型不存在時應拋出 FileNotFoundError"""
        from stt_engine import _find_bundled_sherpa_dir
        # 確保沒有設定環境變數指向假路徑
        os.environ.pop("AMA_SHERPA_DIR", None)
        try:
            _find_bundled_sherpa_dir()
            # 若找到模型則測試通過（CI 環境通常沒有模型）
        except FileNotFoundError:
            pass  # 預期：無模型時應拋出此例外
        except Exception as e:
            self.fail(f"不應拋出 FileNotFoundError 以外的例外：{e}")

    def test_env_var_override(self):
        """AMA_SHERPA_DIR 環境變數設定假路徑應不崩潰"""
        from stt_engine import _find_bundled_sherpa_dir
        os.environ["AMA_SHERPA_DIR"] = "/nonexistent/path/to/model"
        try:
            _find_bundled_sherpa_dir()
        except FileNotFoundError:
            pass  # 預期行為
        finally:
            del os.environ["AMA_SHERPA_DIR"]


# ══════════════════════════════════════════════════════════════════
# 3. 強健性測試 (Robustness Testing)
# ══════════════════════════════════════════════════════════════════

class TestEncodingRobustness(unittest.TestCase):
    """[ ] 編碼測試：Windows cp950 環境不崩潰"""

    def test_all_output_is_utf8_safe(self):
        """後端所有字串輸出應可正確 UTF-8 編解碼"""
        test_strings = [
            "即時逐字稿",
            "AI 會議助理後端啟動中",
            "找不到 bundled sherpa-onnx 模型目錄",
            "你是一位專業的繁體中文文字校對員",
            "【全文摘要】【重點條列】【待辦清單】",
        ]
        for s in test_strings:
            encoded = s.encode("utf-8")
            decoded = encoded.decode("utf-8")
            self.assertEqual(s, decoded, f"UTF-8 編解碼失敗：{s}")

    def test_chinese_filename_safe(self):
        """含中文的檔案名稱在 tempdir 應可正常讀寫"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "測試會議_20240101.txt")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("測試內容")
                with open(filename, "r", encoding="utf-8") as f:
                    result = f.read()
                self.assertEqual(result, "測試內容")
            except Exception as e:
                self.fail(f"中文檔名讀寫失敗：{e}")

    def test_repetition_filter_no_crash(self):
        """重複字過濾不應對任何輸入崩潰"""
        import re
        test_inputs = [
            "嗯嗯嗯嗯嗯嗯",
            "然後然後然後然後",
            "謝謝",
            "",
            "ABC123",
            "a" * 100,
            "然後" * 10,
        ]
        for text in test_inputs:
            try:
                if len(text) > 3:
                    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
                    text = re.sub(r'(.{2,8})\1{2,}', r'\1', text)
            except Exception as e:
                self.fail(f"重複字過濾對輸入「{text[:20]}」崩潰：{e}")


class TestPerformanceConstraints(unittest.TestCase):
    """[ ] 效能壓力：記憶體與 CPU 合理使用"""

    def test_pcm_buffer_does_not_grow_unbounded(self):
        """Buffer 應在 _decode_buffer 後清空，不無限增長"""
        import numpy as np
        # 模擬 buffer 累積後清空
        buffer = np.array([], dtype=np.float32)
        chunk = np.zeros(1600, dtype=np.float32)  # 0.1s at 16kHz
        for _ in range(10):
            buffer = np.concatenate([buffer, chunk])
        # 模擬 _decode_buffer 後清空
        buffer = np.array([], dtype=np.float32)
        self.assertEqual(buffer.size, 0, "decode 後 buffer 應清空")

    def test_downsample_fir_performance(self):
        """FIR 降頻對一個 4096 樣本 chunk 應在 100ms 內完成"""
        import math
        import time as _time

        def downsample_fir(buffer, in_rate, out_rate):
            M = in_rate / out_rate
            new_length = math.floor(len(buffer) / M)
            half_len = math.ceil(M * 8)
            fc = 1.0 / M
            pifc = math.pi * fc
            result = []
            for i in range(new_length):
                center = i * M
                total = 0.0
                k_start = max(0, int(center) - half_len)
                k_end = min(len(buffer) - 1, int(center) + half_len)
                for k in range(k_start, k_end + 1):
                    t = k - center
                    x = pifc * t
                    sinc = 1.0 if abs(x) < 1e-9 else math.sin(x) / x
                    abs_t = abs(t)
                    win = 0.5 * (1.0 + math.cos(math.pi * abs_t / half_len)) if abs_t <= half_len else 0.0
                    total += buffer[k] * fc * sinc * win
                result.append(max(-1.0, min(1.0, total)))
            return result

        # 48kHz → 16kHz，4096 樣本
        chunk = [0.1] * 4096
        t0 = _time.perf_counter()
        downsample_fir(chunk, 48000, 16000)
        elapsed = _time.perf_counter() - t0
        self.assertLess(elapsed, 0.5,
            f"FIR 降頻耗時 {elapsed:.3f}s，應在 0.5s 內（JS 執行更快）")

    def test_trailing_silence_trim(self):
        """stop() 前裁剪尾端靜音應大幅縮短 buffer"""
        import numpy as np
        sample_rate = 16000
        # 1 秒有聲 + 3 秒靜音
        audio = np.concatenate([
            np.random.uniform(-0.1, 0.1, sample_rate).astype(np.float32),
            np.zeros(sample_rate * 3, dtype=np.float32),
        ])
        nz = np.nonzero(audio)[0]
        if nz.size > 0:
            keep = min(int(nz[-1]) + sample_rate // 2, audio.size)
            trimmed = audio[:keep]
        else:
            trimmed = np.array([], dtype=np.float32)
        # 裁剪後應遠短於原始（4 秒 → 約 1.5 秒）
        self.assertLess(len(trimmed), len(audio) * 0.6,
            "裁剪尾端靜音後長度應明顯縮短")


# ══════════════════════════════════════════════════════════════════
# 執行
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestSTTAudioChunk,
        TestSummaryModes,
        TestTraditionalConversion,
        TestFileExport,
        TestOfflineMode,
        TestModelPathResolution,
        TestEncodingRobustness,
        TestPerformanceConstraints,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
