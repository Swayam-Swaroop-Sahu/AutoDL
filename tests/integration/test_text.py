"""Integration test for text end-to-end on messy data (Phase 1f)."""
import os
import tempfile

import pytest


def _make_messy_text_file(path, n=200, malformed_frac=0.10, seed=0):
    """200-line TXT with label<TAB>text including Chinese + Arabic, plus 10% malformed."""
    import numpy as np
    rng = np.random.RandomState(seed)
    chinese = ["很好", "很差", "我喜欢", "讨厌", "一般", "完美", "满意", "糟糕", "惊人", "无感"]
    arabic = ["ممتاز", "سيئ", "رائع", "فظيع", "عادي", "جميل", "مقبول", "بسيط", "متوسط", "رديء"]
    lines = []
    n_malformed = int(n * malformed_frac)
    n_valid = n - n_malformed
    # Valid lines
    for i in range(n_valid):
        label = rng.choice(["pos", "neg"])
        text_parts = []
        for _ in range(rng.randint(3, 8)):
            text_parts.append(rng.choice(chinese + arabic))
        text = " ".join(text_parts)
        lines.append(f"{label}\t{text}")
    # Inject malformed lines (no tab/commma, no label)
    for i in range(n_malformed):
        if rng.random() < 0.5:
            lines.append("just_text_no_label")  # missing label/text split
        else:
            lines.append("")  # blank line
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_text_training_with_arabic_chinese_completes():
    """Non-ASCII text should be preserved and training should complete."""
    from src.preprocessing.text_preprocessor import TextPreprocessor
    from src.data.text_loader import load_text_file, parse_labelled_text

    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "messy.txt")
        _make_messy_text_file(txt)

        lines = load_text_file(txt, min_lines=10)
        assert len(lines) > 0

        texts, labels = parse_labelled_text(lines)
        assert texts is not None and labels is not None
        # Verify Chinese characters survive parsing
        all_text = " ".join(texts)
        assert "很好" in all_text or "ممتاز" in all_text, (
            "Non-ASCII text should survive preprocessing"
        )

        tp = TextPreprocessor(max_words=10000, max_len=40)
        X, y = tp.fit_transform(texts, labels)
        assert X.shape[0] == len(texts)
        assert X.shape[1] == 40
        assert set(y.tolist()) >= {0, 1}


def test_text_predict_preserves_non_ascii():
    """Transforming a Chinese/Arabic string returns the right shape."""
    from src.preprocessing.text_preprocessor import TextPreprocessor
    tp = TextPreprocessor(max_words=10000, max_len=20)
    tp.fit_transform(["hello world", "good day"], ["a", "b"])
    out = tp.transform(["你好世界"])
    assert out.shape == (1, 20)
