# tests/conftest.py
import io
import pytest
import requests
from PIL import Image as PILImage
import torch
from pytest_lazy_fixtures import lf

from viper.src import api


IMG_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"


@pytest.fixture(scope="session")
def raw_image() -> PILImage.Image:
    """Download the demo image once for the whole test session."""
    resp = requests.get(IMG_URL, stream=True)
    resp.raise_for_status()
    img = PILImage.open(io.BytesIO(resp.content)).convert("RGBA")
    return img


@pytest.fixture(scope="session")
def full_patch(raw_image):
    """Whole-image ImagePatch instance."""
    return api.ImagePatch(raw_image)


@pytest.fixture(scope="session")
def alpha_mask(full_patch: "api.ImagePatch"):
    """1xHxW torch alpha mask filled with ones (fully opaque)."""
    _, h, w = full_patch.cropped_image.shape
    return torch.ones(1, h, w)


@pytest.fixture(autouse=True)
def patch_forward_llm_and_simple(monkeypatch):
    """Monkeypatch api.forward ONLY for llm_query/simple_query to avoid real LLM/QA calls."""
    original_forward = api.forward

    def fake_forward(model_name, *args, **kwargs):
        # simple_query uses task='qa'
        if kwargs.get("task") == "qa":
            return "yes"
        # llm_query (and similar) use model names like '<llm>_general' or '<llm>_qa'
        if isinstance(model_name, str) and ("_general" in model_name or "_qa" in model_name or "gpt4" in model_name):
            return "yes"
        # fall back to real forward for everything else
        return original_forward(model_name, *args, **kwargs)

    monkeypatch.setattr(api, "forward", fake_forward)
    yield


@pytest.fixture(scope="session")
def tiny_crop(full_patch):
    return full_patch.crop(10, 10, full_patch.width - 10, full_patch.height - 10)


# Mark to easily deselect the expensive tests if needed: -m "not heavy"
heavy = pytest.mark.heavy


# ---------------------------------------------------------------------------
# Helper assertion utilities
# ---------------------------------------------------------------------------

def assert_imagepatch(obj):
    assert isinstance(obj, api.ImagePatch)


def assert_tensor_image(t):
    import torch
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 3


# tests/test_imagepatch_class.py

def test_imagepatch_from_image_types(full_patch):
    assert_imagepatch(full_patch)
    assert full_patch.alpha is None or isinstance(full_patch.alpha, torch.Tensor)


def test_imagepatch_properties_types(full_patch):
    assert isinstance(full_patch.width, int)
    assert isinstance(full_patch.height, int)
    assert isinstance(full_patch.bbox, tuple) and len(full_patch.bbox) == 4
    assert isinstance(full_patch.horizontal_center, float)
    assert isinstance(full_patch.vertical_center, float)
    assert isinstance(repr(full_patch), str)


def test_imagepatch_to_pil_image(full_patch):
    pil_img = full_patch.to_pil_image()
    from PIL.Image import Image as PILImageType
    assert isinstance(pil_img, PILImageType)


def test_imagepatch_crop_and_alpha(full_patch, alpha_mask):
    cropped = full_patch.crop(5, 5, 5, 5)
    assert_imagepatch(cropped)
    patched = cropped.apply_mask(alpha_mask[:, :cropped.height, :cropped.width])
    assert_imagepatch(patched)
    assert patched.alpha is not None



# tests/test_function_signatures.py


@pytest.mark.parametrize(
    "fn, expected_type, args, kwargs",
    [
        # ImagePatch instance methods
        (lambda patch, objs, **kw: patch.find(objs, **kw), list, (lf("full_patch"), ["person"]), {"mask": False}),
        (lambda patch, name: patch.exists(name), bool, (lf("full_patch"), "person"), {}),
        (lambda patch, category: patch._score(category), float, (lf("full_patch"), "person"), {}),
        (lambda patch, cat, thresh: patch._detect(cat, thresh), bool, (lf("full_patch"), "anything", -1.0), {}),
        (lambda patch, obj, attr: patch.verify_property(obj, attr), bool, (lf("full_patch"), "person", "standing"), {}),
        (lambda patch, options: patch.best_text_match(options), str, (lf("full_patch"), ["cat", "dog", "person"]), {}),
        (lambda patch: patch.compute_depth(), float, (lf("full_patch"),), {}),
        (lambda patch, l, u, r, d: patch.overlaps_with(l, u, r, d), bool, (lf("full_patch"), 0, 0, 10, 10), {}),
        # Module-level helpers
        (api.process_guesses, str, ("Prompt?", None, None), {}),
        (api.best_image_match, (api.ImagePatch, type(None)), ([lf("full_patch")], ["content"]), {}),
        (api.distance, float, (lf("full_patch"), lf("full_patch")), {}),
        (api.bool_to_yesno, str, (True,), {}),
    ],
)
def test_return_types_general(fn, expected_type, args, kwargs):
    res = fn(*args, **kwargs)
    if isinstance(expected_type, tuple):
        assert isinstance(res, expected_type)
    else:
        assert isinstance(res, expected_type)



# tests/test_text_calls.py

def test_simple_query_monkeypatched(full_patch, patch_forward_llm_and_simple):
    res = full_patch.simple_query("What is in the image?")
    assert isinstance(res, str)


def test_llm_query_monkeypatched(patch_forward_llm_and_simple):
    res = api.llm_query("Say hi", long_answer=False)
    assert isinstance(res, str)


# tests/test_lists_and_patches.py

def test_find_output_structure(full_patch):
    out = full_patch.find(["person", "dog"], mask=True)
    assert isinstance(out, list)
    for item in out:
        assert_imagepatch(item)


def test_best_image_match_returns_patch_or_none(full_patch):
    patches = [full_patch]
    selected = api.best_image_match(patches, ["something"])
    assert selected is None or isinstance(selected, type(full_patch))


# tests/test_distance_and_overlap.py

def test_distance_between_patches(full_patch, tiny_crop):
    d = api.distance(full_patch, tiny_crop)
    assert isinstance(d, float)


def test_distance_between_scalars():
    assert isinstance(api.distance(1.0, 5.0), float)


def test_overlaps_with_bool(full_patch):
    assert isinstance(full_patch.overlaps_with(0, 0, full_patch.width, full_patch.height), bool)


# tests/test_bool_helpers.py

def test_bool_to_yesno_values():
    assert api.bool_to_yesno(True) == "yes"
    assert api.bool_to_yesno(False) == "no"



@heavy
@pytest.mark.parametrize("category", ["person", "cat", "dog"])
def test_score_and_detect(full_patch, category):
    s = full_patch._score(category)
    assert isinstance(s, float)
    assert isinstance(full_patch._detect(category, thresh=-1.0), bool)


# Ensure llm_query/process_guesses return str (already tested), but also exercise long_answer branch
@heavy
def test_llm_query_long_answer():
    out = api.llm_query("Explain something briefly", long_answer=True)
    assert isinstance(out, str)


@heavy
def test_process_guesses_three_args():
    out = api.process_guesses("Who am I?", "alice", "bob")
    assert isinstance(out, str)
