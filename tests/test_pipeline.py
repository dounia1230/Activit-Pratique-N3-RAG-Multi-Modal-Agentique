import base64
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from PIL import Image

from rag_pipeline import RAGPipeline, decouper_texte, generer_reponse, image_vers_data_url, recuperer_sources


class CharacterTokenizer:
    def encode(self, texte):
        return list(texte)

    def decode(self, tokens):
        return "".join(tokens)


class RecordingClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="Réponse [S1]"))


def source(kind="texte", contenu="Passage du PDF", chemin="missing.png", reference="S1"):
    return {"reference": reference, "type": kind, "pdf": "exemple.pdf", "page": 1,
            "contenu": contenu, "source": str(chemin)}


def test_generation_sends_only_usable_sources_and_raw_image_base64(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (20, 10), "red").save(image)
    client = RecordingClient()
    sources = [source(), source("image", chemin=image, reference="S2"),
               source("image", chemin=image, reference="S3"), source(contenu=" ")]
    reponse, transmises, warnings = generer_reponse("Question ?", sources, client, "local", max_images=1)
    assert reponse == "Réponse [S1]"
    assert [s["reference"] for s in transmises] == ["S1", "S2"]
    assert len(warnings) == 1
    message = client.calls[0]["messages"][1]
    assert base64.b64decode(message["images"][0]).startswith(b"\x89PNG")
    assert not message["images"][0].startswith("data:")
    assert '"reference": "S2"' in message["content"]


def test_generation_skips_missing_images_and_never_calls_llm_without_sources():
    client = RecordingClient()
    _, transmises, warnings = generer_reponse("Question ?", [source("image")], client, "local")
    assert transmises == [] and len(warnings) == 1 and client.calls == []


@pytest.mark.parametrize("value", [-1, True, 1.5, 13])
def test_generation_rejects_invalid_image_limit(value):
    with pytest.raises(ValueError):
        generer_reponse("Question ?", [], RecordingClient(), "local", value)


def test_image_conversion_preserves_original(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGBA", (200, 100)).save(image)
    original = image.read_bytes()
    assert image_vers_data_url(image, (20, 20)).startswith("data:image/png;base64,")
    assert image.read_bytes() == original


def test_chunk_overlap_and_invalid_arguments():
    bpe = CharacterTokenizer()
    assert list(decouper_texte("abcdefghij", bpe, 4, 1)) == ["abcd", "defg", "ghij"]
    assert list(decouper_texte("", bpe)) == []
    with pytest.raises(ValueError):
        list(decouper_texte("abc", bpe, 4, 4))


def test_pdf_index_is_idempotent_and_separates_documents(tmp_path, monkeypatch):
    pipeline = RAGPipeline(tmp_path)
    monkeypatch.setattr(pipeline, "charger_clip", lambda: None)
    pipeline.bpe = CharacterTokenizer()
    monkeypatch.setattr(pipeline, "encoder_textes", lambda textes: [[1.0, 0.0] for _ in textes])
    monkeypatch.setattr(pipeline, "encoder_image", lambda image: [0.0, 1.0])
    image = tmp_path / "figure.png"
    Image.new("RGB", (20, 20), "blue").save(image)
    for name, text in [("premier.pdf", "Transformer attention"), ("second.pdf", "Position encoding")]:
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), text)
            page.insert_image(pymupdf.Rect(72, 100, 172, 200), filename=str(image))
            pdf.save(tmp_path / name)
    first = pipeline.indexer_pdf(tmp_path / "premier.pdf")
    count = pipeline.collection.count()
    again = pipeline.indexer_pdf(tmp_path / "premier.pdf")
    assert again == first and pipeline.collection.count() == count
    second = pipeline.indexer_pdf(tmp_path / "second.pdf")
    assert first["document_id"] != second["document_id"]
    assert pipeline.collection.count() == count * 2
    assert first["images"] == 1
    results = pipeline.rechercher(texte="attention", type_resultat="texte", k=12)
    sources = recuperer_sources(results)
    assert all(s["type"] == "texte" and s["page"] == 1 for s in sources)
    assert {s["pdf"] for s in sources} == {"premier.pdf", "second.pdf"}
    assert all(Path(s["source"]).is_file() for s in sources)
    images = pipeline.rechercher(image=image, type_resultat="image", k=2)
    assert all(m["type"] == "image" for m in images["metadatas"][0])


@pytest.mark.parametrize("kwargs", [
    {}, {"texte": "x", "image": "x.png"}, {"texte": " "}, {"texte": 4},
    {"texte": "x", "k": True}, {"texte": "x", "k": 13}, {"texte": "x", "type_resultat": "bad"},
])
def test_invalid_query_is_rejected_before_loading_models(kwargs):
    with pytest.raises(ValueError):
        RAGPipeline().rechercher(**kwargs)


def test_empty_collection_does_not_load_clip(tmp_path, monkeypatch):
    pipeline = RAGPipeline(tmp_path)
    monkeypatch.setattr(pipeline, "charger_clip", lambda: pytest.fail("CLIP must remain unloaded"))
    with pytest.raises(ValueError, match="vide"):
        pipeline.rechercher(texte="question")


def test_notebook_has_documented_valid_cells_without_outputs():
    import ast
    import nbformat
    notebook = nbformat.read("rag_multimodal_agentic.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, c in enumerate(notebook.cells):
        if c.cell_type != "code":
            continue
        assert c.source.strip(), f"Empty code cell {index}"
        ast.parse(c.source)
        assert c.outputs == []
        assert index > 0 and notebook.cells[index - 1].cell_type == "markdown"
