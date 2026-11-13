from io import BytesIO
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_renders_without_loading_models():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Interroger vos PDF"
    next(button for button in app.button if button.label == "Rechercher et répondre").click().run()
    assert not app.exception
    assert "question" in app.info[0].value


def test_upload_name_cannot_escape_upload_directory(tmp_path):
    from app import sauvegarder_upload
    upload = BytesIO(b"document")
    upload.name = "../../outside.pdf"
    chemin = sauvegarder_upload(upload, tmp_path)
    assert chemin.resolve().is_relative_to(tmp_path.resolve())
    assert chemin.name == "outside.pdf" and chemin.read_bytes() == b"document"
