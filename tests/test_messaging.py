"""Message checks and the write/score/retry graph, with fake models (no network)."""
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from messaging import agent, checks

FACTS = {
    "user_id": 1,
    "facts": "- Recommended benefit: 3% cash-back on dining, up to €30/month (premium card)\n"
             "- Their dining spend: €1,250/month\n- Value to them: €30/month (€360/year)\n"
             "- Premium card fee: €240/year",
    "persona": "- tone: friendly\n- motivation: savings\n- digital: high\n- channel: email",
    "channel": "email", "language": "en", "language_name": "English",
}
GOOD = ("Hi there,\n\nYou spend €1,250 a month on dining. With 3% cash-back on dining you could earn €30 a month, "
        "that's €360 a year.\n\nBest,\nClaire")
JUDGE_OK = json.dumps({k: {"score": 5, "reason": "fine"} for k in agent.JUDGE_CRITERIA})


def failed(msg, facts=FACTS):
    return {k for k, (ok, _) in checks.run(msg, facts).items() if not ok}


def test_checks():
    assert failed(GOOD) == set()
    assert failed(GOOD.replace("€30 a month", "30 EUR a month")) == {"currency"}
    assert failed(GOOD.split("\n", 1)[1]) == {"greeting"}
    assert failed(GOOD.replace("Claire", "Bob")) == {"signature"}
    assert failed(GOOD.replace("€30 a month", "€45 a month")) == {"numbers"}
    assert failed(GOOD + " " + "x" * 300, FACTS | {"channel": "sms"}) == {"length"}
    assert checks.parse_numbers("1 250,50 € et 3,5 % ou 1,234.5") == [1250.5, 3.5, 1234.5]
    fr = FACTS | {"language": "fr", "language_name": "French"}
    assert failed("Bonjour,\nVous dépensez 1 250 € par mois.\nClaire", fr) == set()


def test_graph_approves_good_draft():
    out = agent.write_message(FACTS, FakeListChatModel(responses=[GOOD]), FakeListChatModel(responses=[JUDGE_OK]))
    assert out["approved"] and not out["flagged"] and out["attempts"] == 1 and out["draft"] == GOOD


def test_graph_retries_twice_then_flags():
    bad = GOOD.replace("Claire", "The bank")
    out = agent.write_message(FACTS, FakeListChatModel(responses=[bad] * 3), FakeListChatModel(responses=[JUDGE_OK] * 3))
    assert out["flagged"] and not out["approved"] and out["draft"] is None
    assert out["attempts"] == 3 and len(out["scores"]) == 3 and not out["scores"][-1]["signature"]["passed"]
    assert "previous draft was rejected" in out["prompt"] and "signature" in out["prompt"]


def test_graph_recovers_on_retry_and_bad_judge_json_fails():
    out = agent.write_message(FACTS, FakeListChatModel(responses=[GOOD.replace("€", "EUR "), GOOD]),
                              FakeListChatModel(responses=["not json", JUDGE_OK]))
    assert out["approved"] and out["attempts"] == 2 and "judge" in out["scores"][0]
