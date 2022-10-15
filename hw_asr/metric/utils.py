import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    splitted_target_text: str = target_text.split(' ')
    if len(splitted_target_text) == 0:
        return 1
    return editdistance.distance(splitted_target_text, predicted_text.split(' ')) / len(splitted_target_text)
