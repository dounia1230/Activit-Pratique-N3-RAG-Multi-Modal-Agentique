import json

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image

from rag_agent import analyser_execution, creer_outil, message_utilisateur


def test_tool_returns_sources_without_local_paths():
    def rag(**kwargs):
        assert kwargs["type_resultat"] is None
        return {"question": kwargs["question"], "reponse": "Réponse",
                "sources_transmises": [{"reference": "S1", "id": "a", "pdf": "a.pdf", "page": 3,
                                        "type": "image", "source": "C:/private/image.png", "contenu": "x"}],
                "avertissements": []}
    outil = creer_outil(rag)
    contenu = outil.invoke({"question": "Question"})
    assert "private" not in contenu and '"page": 3' in contenu
    with pytest.raises(ValueError):
        outil.invoke({"question": "Question", "k": 13})


def test_agent_image_input_and_empty_input(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (20, 20)).save(image)
    message = message_utilisateur(image=image, type_resultat="image", k=3)
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "k=3" in message["content"][0]["text"]
    with pytest.raises(ValueError):
        message_utilisateur(texte=" ")
    with pytest.raises(ValueError):
        message_utilisateur(texte="Question", k=True)


def test_agent_preserves_call_local_citations_and_deduplicates_images(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (20, 20), "red").save(image)
    class Collection:
        def get(self, **kwargs):
            return {"metadatas": [{"pdf": "a.pdf", "page": 3, "source": str(image)}]}
    retour = json.dumps({"sources_transmises": [
        {"reference": "S1", "id": "image1", "type": "image", "pdf": "a.pdf", "page": 3}],
        "avertissements": []})
    messages = [ToolMessage(content=retour, name="consulter_pdf", tool_call_id=identifiant)
                for identifiant in ["appel1", "appel2"]]
    messages.append(AIMessage(content=[{"type": "text", "text": "Synthèse"}]))
    sortie = analyser_execution({"messages": messages}, Collection())
    assert sortie["texte"] == "Synthèse"
    assert len(sortie["images"]) == 1
    assert [s["appel_id"] for s in sortie["sources"]] == ["appel1", "appel2"]
    assert analyser_execution({"messages": messages}, Collection(), "texte")["images"] == []


def test_malformed_tool_return_becomes_warning():
    sortie = analyser_execution({"messages": [
        ToolMessage(content="broken JSON", name="consulter_pdf", tool_call_id="a"),
        AIMessage(content="Réponse"),
    ]})
    assert len(sortie["avertissements"]) == 1


class LoopingChatModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "looping-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="", tool_calls=[{
            "name": "consulter_pdf", "args": {"question": "Question"},
            "id": f"call-{len(messages)}", "type": "tool_call",
        }])
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_agent_enforces_three_tool_calls_even_when_model_keeps_requesting():
    appels = []
    def rag(**kwargs):
        appels.append(kwargs)
        return {"question": kwargs["question"], "reponse": "Réponse", "sources_transmises": [], "avertissements": []}
    agent = create_agent(LoopingChatModel(), tools=[creer_outil(rag)],
                         middleware=[ToolCallLimitMiddleware(run_limit=3, exit_behavior="end")])
    execution = agent.invoke({"messages": [{"role": "user", "content": "Question"}]},
                             config={"recursion_limit": 20})
    assert len(appels) == 3
    assert execution["messages"][-1].content
