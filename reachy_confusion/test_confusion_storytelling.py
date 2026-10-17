"""Fast logic tests, without a Reachy Mini or OpenAI key."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from confusion_storytelling import parse_label, explicit_request, classify, mock_predict


class ParsingTests(unittest.TestCase):
    def test_correct_labels(self):
        for label in ["CONFUSED", "NOT_CONFUSED", "UNCERTAIN"]:
            self.assertEqual(parse_label('{"label":"' + label + '"}'), label)

    def test_malformed_fails_safe(self):
        self.assertEqual(parse_label("The user seems confused."), "UNCERTAIN")
        self.assertEqual(parse_label('{"label":"MAYBE"}'), "UNCERTAIN")
        self.assertEqual(parse_label('{"label":"NOT_CONFUSED"}'), "NOT_CONFUSED")

    def test_explicit_help(self):
        self.assertTrue(explicit_request("Could you clarify what happened?"))
        self.assertFalse(explicit_request("They should open the door."))

    def test_prompt_path(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"NOT_CONFUSED"}'))]
        )
        result = classify(client, "gpt-4.1-mini", "I think they should go outside", "Mira had a key")
        self.assertEqual(result.label, "NOT_CONFUSED")
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-4.1-mini")
        self.assertEqual(sent["messages"][1]["content"].count("Mira had a key"), 1)

    def test_mock_is_explicitly_mock(self):
        prediction = mock_predict("Wait, I don't understand")
        self.assertEqual(prediction.label, "CONFUSED")
        self.assertIn("MOCK", prediction.error)


if __name__ == "__main__":
    unittest.main()
