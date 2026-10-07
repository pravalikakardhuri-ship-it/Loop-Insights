"""
tests/test_catalog.py
-----------------------
The product catalog in config.py feeds the sample data generator and the
Manual Feedback form, so keep it consistent.
"""

import importlib.util
from pathlib import Path

from config import config


def _load_generator():
    path = Path(__file__).resolve().parent.parent / "data" / "generate_sample_data.py"
    spec = importlib.util.spec_from_file_location("generate_sample_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_has_no_duplicate_products():
    products = [p for plist in config.PRODUCT_CATALOG.values() for p in plist]
    assert len(products) == len(set(products))
    assert len(products) >= 20


def test_product_category_map_covers_catalog():
    mapping = config.product_category_map()
    for category, products in config.PRODUCT_CATALOG.items():
        for product in products:
            assert mapping[product] == category


def test_generated_data_covers_every_product_and_category():
    gen = _load_generator()
    rows = gen.generate_dataset(2000)
    assert {r["product"] for r in rows} == set(config.product_category_map())
    assert {r["category"] for r in rows} == set(config.PRODUCT_CATALOG)
    for r in rows:
        assert config.product_category_map()[r["product"]] == r["category"]


def test_electronics_only_phrases_are_not_used_for_other_products():
    gen = _load_generator()
    # Words that make no sense for shoes, perfume, etc. ("noise" is left out:
    # it is realistic for appliances and comes from their own templates).
    electronics_words = ("battery", "display", "overheat", "gadget", "screen", "interface")
    rows = gen.generate_dataset(3000)
    for r in rows:
        if r["category"] in gen.ELECTRONICS_CATEGORIES:
            continue
        text = r["feedback"].lower()
        assert not any(w in text for w in electronics_words), text


def test_branding_comes_from_config():
    assert config.APP_TITLE
    assert config.APP_TAGLINE == "Listen. Understand. Improve."
