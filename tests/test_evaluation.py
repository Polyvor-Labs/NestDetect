from nestdetect.evaluation import forgetting_score


def test_forgetting_score_keeps_sign() -> None:
    assert forgetting_score(0.82, 0.74) == 0.08
    assert forgetting_score(0.70, 0.75) == -0.05
