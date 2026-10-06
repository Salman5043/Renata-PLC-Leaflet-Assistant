from app.rag_graph import ABSTAIN, citation_labels_valid, detect_product_filter, detect_section_intent, lexical_overlap

def test_product_detection():
    assert detect_product_filter("What is the dose of Rolac?") == "Rolac 10 mg"
    assert detect_product_filter("Tell me about Doxicap 100 mg") == "Doxicap 100 mg"

def test_section_detection():
    assert detect_section_intent("Use in pregnancy and breast-feeding") == "pregnancy"
    assert detect_section_intent("What is Maxpro used for?") == "what"

def test_lexical_overlap():
    assert lexical_overlap("Rolac dose", "Rolac dose is 10 mg every 4-6 hours") > 0

def test_citation_validation():
    docs = [{"source":"a"},{"source":"b"}]
    assert citation_labels_valid("Answer [C1][C2].", docs)
    assert not citation_labels_valid("Answer [C3].", docs)
    assert not citation_labels_valid("Answer without citation.", docs)

def test_abstain_constant():
    assert "provided Renata leaflets" in ABSTAIN
