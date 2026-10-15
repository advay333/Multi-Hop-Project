from mhtable.env.text import LETTER, QUOTED, LexicalSimilarity, dates, name_similarity, numbers, op_cues, tokens


def test_tokens_split_and_fold_plurals():
    assert tokens("customerID of Categories") == ("customer", "id", "category")
    assert tokens("class") == ("class",)


def test_numbers_digits_words_and_sentence_end():
    assert numbers("price more than 4,000,000.") == [4000000]
    assert numbers("third grade, aged 32 or older") == [32, 3]
    assert numbers("rated 3.5 or 10") == [3.5, 10]


def test_operator_cues():
    assert op_cues("pilots younger than 34") == {"<"}
    assert op_cues("aged 32 or older") == {">="}
    assert op_cues("trips that did not end in San Francisco") == {"!="}
    assert op_cues("the name of the artist") == set()


def test_dates_quotes_letters():
    assert dates("between 1975-01-01 and 1976-01-01??") == ["1975-01-01", "1976-01-01"]
    assert QUOTED.findall("whose name has 'Led'") == ["Led"]
    assert LETTER.findall("contain the letter s in their names") == ["s"]


def test_similarity_ranks_related_names_higher():
    sim = LexicalSimilarity()
    assert sim.score("heads born in Alabama", "born_state") > sim.score("heads born in Alabama", "budget")
    assert name_similarity("customer_id", "customers id") > name_similarity("customer_id", "artists id")
