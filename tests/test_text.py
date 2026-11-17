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


def test_explicit_filters_from_bird_evidence():
    from mhtable.env.text import explicit_filters
    ev = ("refers to WasCompiled = 0; IndepYear BETWEEN 1980 AND 1995; courseLevel = 'Level_400'; "
          "course.course_id < 10; `customer id` <> 'AB-1'; Region IN ('East', 'West'); "
          "name LIKE '%Ale%'; T1.x IS NOT NULL; price >= 3.5")
    assert explicit_filters(ev) == [
        ("wascompiled", "=", (0,)), ("indepyear", "between", (1980, 1995)),
        ("courselevel", "=", ("Level_400",)), ("course.course_id", "<", (10,)),
        ("customer id", "!=", ("AB-1",)), ("region", "in", ("East", "West")),
        ("name", "like", ("%Ale%",)), ("t1.x", "not_is", ()), ("price", ">=", (3.5,))]
    assert explicit_filters("How many people are older than 30?") == []


def test_date_prefixes_numbers_glued_ids_and_contexts():
    from mhtable.env.text import date_prefixes, number_contexts
    assert date_prefixes("movies released in 2006") == ["2006"]
    assert date_prefixes("shipped in February 2017") == ["2017-02", "2017"]
    assert date_prefixes("price 2006.5, id 12006") == []
    assert numbers("menu ID12882 and object sample no.14") == [12882, 14]
    ctx = number_contexts("Which platform is game ID 178 on? A repository of 254 likes, 88 followers.")
    assert ctx[178] == "platform game id" and ctx[254] == "repository likes" and ctx[88] == "followers"
