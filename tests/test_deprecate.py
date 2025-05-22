from ssms.config import DeprecatedDict


def test_deprecated_dict_warns(recwarn):
    d = DeprecatedDict(lambda x: x, "replacement")
    val = d["foo"]
    w = recwarn.pop(DeprecationWarning)
    message = str(w.message)
    assert "deprecated" in message
    assert "replacement" in message
    assert val == "foo"
