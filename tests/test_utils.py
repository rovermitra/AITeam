import pytest
#from utils import parse_llm_json
from utils import parse_llm_json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import parse_llm_json
import pytest

def test_parse_clean_json():
    s = '{"matches":[{"name":"A","explanation":"Good","compatibility_score":0.9}]}'
    result = parse_llm_json(s)
    assert isinstance(result, list)
    assert result[0]['name'] == "A"

def test_parse_wrapped_json():
    s = '{"result": [{"name":"A"}]}'
    result = parse_llm_json(s)
    assert isinstance(result, list)
    assert result[0]['name'] == "A"

def test_parse_json_in_text():
    s = "Some intro text\n[{\"name\":\"A\"}] more text"
    result = parse_llm_json(s)
    assert isinstance(result, list)
    assert result[0]['name'] == "A"

def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_llm_json("No JSON here")

