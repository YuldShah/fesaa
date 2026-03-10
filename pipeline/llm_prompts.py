IELTS_RUBRIC_XML = """<ielts_rubric>
  <criterion name="Fluency and Coherence">
    <band score="9">Speaks fluently with only rare repetition or self correction; any hesitation is content-related rather than to find words or grammar. Speaks coherently with fully appropriate cohesive features. Develops topics fully and appropriately.</band>
    <band score="8">Speaks fluently with only occasional repetition or self-correction; hesitation is usually content-related and only rarely to search for language. Develops topics coherently and appropriately.</band>
    <band score="7">Speaks at length without noticeable effort or loss of coherence. May demonstrate language-related hesitation at times, or some repetition and/or self-correction. Uses a range of connectives and discourse markers with some flexibility.</band>
    <band score="6">Is willing to speak at length, though may lose coherence at times due to occasional repetition, self-correction or hesitation. Uses a range of connectives and discourse markers but not always appropriately.</band>
    <band score="5">Usually maintains flow of speech but uses repetition, self-correction and/or slow speech to keep going. May over-use certain connectives and discourse markers. Produces simple speech fluently, but more complex communication causes fluency problems.</band>
    <band score="4">Cannot respond without noticeable pauses and may speak slowly, with frequent repetition and self-correction. Links basic sentences but with repetitious use of simple connectives and some breakdowns in coherence.</band>
    <band score="3">Speaks with long pauses. Has limited ability to link simple sentences. Gives only simple responses and is frequently unable to convey basic message.</band>
  </criterion>
  <criterion name="Lexical Resource">
    <band score="9">Uses vocabulary with full flexibility and precision in all topics. Uses idiomatic language naturally and accurately.</band>
    <band score="8">Uses a wide vocabulary resource readily and flexibly to convey precise meaning. Uses less common and idiomatic vocabulary skilfully, with occasional inaccuracies. Uses paraphrase effectively as required.</band>
    <band score="7">Uses vocabulary resource flexibly to discuss a variety of topics. Uses some less common and idiomatic vocabulary and shows some awareness of style and collocation, with some inappropriate choices. Uses paraphrase effectively.</band>
    <band score="6">Has a wide enough vocabulary to discuss topics at length and make meaning clear in spite of inappropriacies. Generally paraphrases successfully.</band>
    <band score="5">Manages to talk about familiar and unfamiliar topics but uses vocabulary with limited flexibility. Attempts to use paraphrase but with mixed success.</band>
    <band score="4">Is able to talk about familiar topics but can only convey basic meaning on unfamiliar topics and makes frequent errors in word choice. Rarely attempts paraphrase.</band>
    <band score="3">Uses simple vocabulary to convey personal information. Has insufficient vocabulary for less familiar topics.</band>
  </criterion>
  <criterion name="Grammatical Range and Accuracy">
    <band score="9">Uses a full range of structures naturally and appropriately. Produces consistently accurate structures apart from 'slips' characteristic of native speaker speech.</band>
    <band score="8">Uses a wide range of structures flexibly. Produces a majority of error-free sentences with only very occasional inappropriacies or basic/non-systematic errors.</band>
    <band score="7">Uses a range of complex structures with some flexibility. Frequently produces error-free sentences, though some grammatical mistakes persist.</band>
    <band score="6">Uses a mix of simple and complex structures, but with limited flexibility. May make frequent mistakes with complex structures, though these rarely cause comprehension problems.</band>
    <band score="5">Produces basic sentence forms with reasonable accuracy. Uses a limited range of more complex structures, but these usually contain errors and may cause some comprehension problems.</band>
    <band score="4">Produces basic sentence forms and some correct simple sentences but subordinate structures are rare. Errors are frequent and may lead to misunderstanding.</band>
    <band score="3">Attempts basic sentence forms but with limited success, or relies on apparently memorised utterances. Makes numerous errors except in memorised expressions.</band>
  </criterion>
  <criterion name="Pronunciation">
    <band score="9">Uses a full range of pronunciation features with precision and subtlety. Sustains flexible use of features throughout. Is effortless to understand.</band>
    <band score="8">Uses a wide range of pronunciation features. Sustains flexible use of features, with only occasional lapses. Is easy to understand throughout; L1 accent has minimal effect on intelligibility.</band>
    <band score="7">Shows all the positive features of Band 6 and some, but not all, of the positive features of Band 8.</band>
    <band score="6">Uses a range of pronunciation features with mixed control. Shows some effective use of features but this is not sustained. Can generally be understood throughout, though mispronunciation of individual words or sounds reduces clarity at times.</band>
    <band score="5">Shows all the positive features of Band 4 and some, but not all, of the positive features of Band 6.</band>
    <band score="4">Uses a limited range of pronunciation features. Attempts to control features but lapses are frequent. Mispronunciations are frequent and cause some difficulty for the listener.</band>
    <band score="3">Shows some of the features of Band 2 and some, but not all, of the positive features of Band 4.</band>
  </criterion>
</ielts_rubric>"""

GEMINI_RULES = """1. THIS IS AN AUDIO TRANSCRIPT. Evaluate spoken English, not spelling.
2. NEVER penalize grammar for ASR homophone substitutions (it's vs its).
3. Assume nonsensical words that sound similar are ASR errors.
4. Stutters/false starts evaluated EXCLUSIVELY under Fluency.
5. Ignore missing filler tokens dropped by ASR on noisy audio.
6. ACCEPT NATURAL SPOKEN FEATURES (contractions, informal discourse markers).
7. PENALIZE FORCED IDIOMS (raining cats and dogs).
8. DISTINGUISH PAUSING TYPES: Differentiate natural chunking vs language-related hesitation.
9. REWARD GRAMMATICAL RISK-TAKING.
10. QUOTE-AND-MATCH: Explicitly quote the <ielts_rubric> for evidence.
11. NUMERIC SCORES ARE PRE-COMPUTED in fixed_scores. DO NOT generate or alter rubric numbers.
12. LOW-CONFIDENCE SHORT FUNCTION WORDS ARE WEAK EVIDENCE; prioritize intelligibility patterns over isolated flags."""

GEMINI_SYSTEM_PROMPT = f"""You are an expert IELTS Speaking examiner.
Use the official rubric and follow the rules exactly.

{IELTS_RUBRIC_XML}

Apply these exact 12 rules:
{GEMINI_RULES}

Return only valid JSON that matches the required schema.
Do not return numeric scores."""

_DOMAIN_ITEM = {
    "type": "object",
    "required": ["strength", "strength_evidence", "error", "error_evidence", "rubric_justification", "drill"],
    "properties": {
        "strength": {"type": "string"},
        "strength_evidence": {"type": "string"},
        "error": {"type": "string"},
        "error_evidence": {"type": "string"},
        "rubric_justification": {"type": "string"},
        "drill": {"type": "string"},
    },
}

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["summary", "domain_feedback"],
    "properties": {
        "summary": {"type": "string"},
        "domain_feedback": {
            "type": "object",
            "required": ["fluency_coherence", "lexical_resource", "grammatical_range_accuracy", "pronunciation"],
            "properties": {
                "fluency_coherence": _DOMAIN_ITEM,
                "lexical_resource": _DOMAIN_ITEM,
                "grammatical_range_accuracy": _DOMAIN_ITEM,
                "pronunciation": _DOMAIN_ITEM,
            },
        },
        "sample_rewrite": {"type": "string"},
    },
}
