import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "synthetic_queries.txt"


def item(kind, doc, query, answer, *anchors):
    return {
        "query_type": kind,
        "document_key": doc,
        "query": query,
        "expected_answer": answer,
        "anchor_groups": [list(group) for group in anchors],
    }


SPECS = [
    # Direct / lexical (20)
    item("direct_lexical", "100", "What are the four AI RMF core functions?", "The four functions are GOVERN, MAP, MEASURE, and MANAGE.", ("govern", "map", "measure", "manage", "core")),
    item("direct_lexical", "100", "What does the GOVERN function cultivate and implement?", "It cultivates and implements a culture of risk management in organizations working with AI systems.", ("govern function", "culture of risk management")),
    item("direct_lexical", "100", "What does the MAP function establish for framing AI risks?", "MAP establishes the context needed to frame risks related to an AI system.", ("map function", "context", "frame risks")),
    item("direct_lexical", "100", "What is the purpose of the MEASURE function?", "MEASURE employs quantitative, qualitative, or mixed methods to analyze, assess, benchmark, and monitor AI risk and trustworthiness.", ("measure function", "quantitative", "qualitative", "monitor")),
    item("direct_lexical", "100", "What risk response options are listed under MANAGE?", "The options include mitigating, transferring, avoiding, or accepting risk.", ("mitigating", "transferring", "avoiding", "accepting")),
    item("direct_lexical", "100", "Which trustworthy AI characteristic is described as a necessary condition?", "Valid and reliable is described as a necessary condition of trustworthiness.", ("valid", "reliable", "necessary condition")),
    item("direct_lexical", "100", "What does TEVV stand for?", "TEVV stands for test, evaluation, verification, and validation.", ("test", "evaluation", "verification", "validation", "tevv")),
    item("direct_lexical", "100", "Who are affected individuals and communities?", "They are people, groups, communities, or organizations directly or indirectly affected by AI systems or AI-based decisions.", ("affected individuals", "directly or indirectly affected")),
    item("direct_lexical", "100", "What is an AI RMF use-case profile?", "It is an implementation of AI RMF functions, categories, and subcategories for a specific setting or application.", ("use-case profiles", "specific setting", "application")),
    item("direct_lexical", "100", "What does the framework say about risk tolerance?", "Risk tolerance is the organization's or stakeholder's readiness to bear risk in pursuit of objectives.", ("risk tolerance", "readiness", "bear risk")),
    item("direct_lexical", "600", "What is the Generative AI Profile intended to help organizations do?", "It helps organizations identify risks unique to or worsened by generative AI and propose actions for managing them.", ("generative ai profile", "identify", "risks", "actions")),
    item("direct_lexical", "600", "What is confabulation in generative AI?", "Confabulation is the production of confidently stated but erroneous or false content by a generative AI system.", ("confabulation", "erroneous", "false")),
    item("direct_lexical", "600", "What is data provenance?", "Data provenance records the origin, history, transformations, and lineage of data or digital content.", ("data provenance", "origin", "history")),
    item("direct_lexical", "600", "What does the profile recommend tracing for digital content?", "It recommends tracing the origin and modifications of digital content.", ("trace the origin", "modifications", "digital content")),
    item("direct_lexical", "600", "What should be reviewed and verified in GAI system outputs?", "Sources and citations in GAI outputs should be reviewed and verified.", ("review and verify", "sources", "citations")),
    item("direct_lexical", "600", "Which environmental impacts should organizations measure or estimate?", "They should measure or estimate energy and water consumption for training, fine-tuning, and deployment.", ("environmental impacts", "energy", "water consumption")),
    item("direct_lexical", "600", "What is AI red-teaming used to identify?", "It is used to identify potential adverse model behavior or outcomes, how they could occur, and whether safeguards withstand stress.", ("ai red-teaming", "adverse behavior", "stress test")),
    item("direct_lexical", "600", "What does MG-4.2-001 recommend publishing?", "It recommends publishing regular reports describing system performance, feedback received, and improvements made.", ("mg-4.2-001", "publish", "performance", "feedback")),
    item("direct_lexical", "600", "What is model collapse?", "Model collapse is degradation that can occur when models are recursively trained on synthetic or AI-generated data.", ("model collapse", "synthetic", "training")),
    item("direct_lexical", "600", "What are content provenance techniques meant to support?", "They support tracking content origin and history and helping users distinguish authentic from manipulated or AI-generated content.", ("content provenance", "origin", "ai-generated")),

    # Paraphrased / semantic (30)
    item("paraphrased_semantic", "100", "Why should AI risk work begin during planning and design rather than after deployment?", "Starting early allows safety considerations and risk controls to be incorporated throughout the lifecycle instead of reacting only after harm occurs.", ("safety considerations", "lifecycle", "planning and design")),
    item("paraphrased_semantic", "100", "Why can the same AI system create different risks in different deployments?", "Risks depend on context, users, operating conditions, and how a system is used, so a new deployment can differ from the developer's original assumptions.", ("deployment", "context", "risks", "developer")),
    item("paraphrased_semantic", "100", "Why does NIST call AI systems socio-technical?", "Their outcomes arise from interactions among technical components, human behavior, social dynamics, and the deployment context.", ("socio-technical", "human behavior", "social")),
    item("paraphrased_semantic", "100", "How does diverse participation improve AI risk assessment?", "Diverse experience, expertise, and backgrounds broaden the identification of impacts, assumptions, and risks across the AI lifecycle.", ("diversity", "experience", "expertise", "impacts")),
    item("paraphrased_semantic", "100", "Why might a technically accurate AI system still be untrustworthy?", "Trustworthiness also depends on safety, security, accountability, transparency, explainability, privacy, and fairness in its context of use.", ("trustworthy", "characteristics", "context of use")),
    item("paraphrased_semantic", "100", "How can transparency increase confidence in an AI system?", "It gives relevant people information appropriate to their roles so they can understand the system, its outputs, and its limitations.", ("transparency", "confidence", "understanding")),
    item("paraphrased_semantic", "100", "Why must AI risk metrics be selected with deployment context in mind?", "Developer metrics may not align with the methods, priorities, or tolerances of the organization operating the system.", ("metrics", "developing", "deploying", "align")),
    item("paraphrased_semantic", "100", "Why is ongoing monitoring needed after an AI system is released?", "Risks, impacts, data, and system behavior can change, so measurement and management must continue during operation.", ("continue applying", "measure function", "monitoring", "risks")),
    item("paraphrased_semantic", "100", "How can pre-trained models create both benefits and risks?", "They can improve performance and advance research while increasing uncertainty, opacity, bias-management difficulty, and reproducibility concerns.", ("pre-trained models", "bias", "reproducibility")),
    item("paraphrased_semantic", "100", "Why should people validating an AI system be distinct from its builders?", "Separation supports independent scrutiny and reduces conflicts or blind spots in verification and validation.", ("verification", "validation", "distinct", "design")),
    item("paraphrased_semantic", "100", "How does the AI RMF support organizations with different resources?", "It is voluntary, flexible, non-sector-specific, and use-case agnostic, allowing adoption in varying degrees and capacities.", ("voluntary", "non-sector-specific", "use-case agnostic", "flexibility")),
    item("paraphrased_semantic", "100", "Why can human oversight alone fail to eliminate AI bias?", "Human decisions and organizational processes can introduce their own cognitive and systemic biases throughout the lifecycle.", ("human", "cognitive biases", "systemic biases")),
    item("paraphrased_semantic", "100", "What makes failure modes of large pretrained models difficult to anticipate?", "Their scale, complexity, emergent properties, and opaque behavior make failures harder to predict and test comprehensively.", ("failure modes", "emergent properties", "large-scale")),
    item("paraphrased_semantic", "100", "How can privacy-enhancing methods reduce AI data risks?", "Methods such as de-identification, aggregation, and data minimization can limit exposure of personal information.", ("de-identification", "aggregation", "privacy")),
    item("paraphrased_semantic", "100", "Why are AI risk responses not a fixed checklist?", "Functions, categories, and actions are outcome-oriented and should be adapted to the system, organization, and context rather than followed as ordered steps.", ("do not constitute a checklist", "ordered set", "outcomes")),
    item("paraphrased_semantic", "600", "Why can generative models make false information especially persuasive?", "They can produce fluent, confident, human-like content even when claims are erroneous, making mistakes difficult for users to recognize.", ("confabulation", "confidently", "false")),
    item("paraphrased_semantic", "600", "How can synthetic content weaken the information ecosystem?", "Large volumes of realistic generated content can obscure provenance, spread misinformation, enable impersonation, and make authentic material harder to distinguish.", ("information integrity", "synthetic content", "provenance")),
    item("paraphrased_semantic", "600", "Why should a fine-tuned third-party model be reassessed?", "Fine-tuning or adapting it to a new domain can change behavior and risk, so earlier measurements and supplier assumptions may no longer hold.", ("fine-tuned", "third-party models", "reassess risk")),
    item("paraphrased_semantic", "600", "How does involving end users help manage generative AI risk?", "Their feedback reveals real interaction patterns, usability problems, misuse, and impacts that laboratory testing may miss.", ("end-users", "practitioners", "operators", "feedback")),
    item("paraphrased_semantic", "600", "Why should organizations retain access to an untuned baseline model?", "A baseline helps debug and compare the effects of fine-tuning, added data, parameter changes, and other modifications.", ("un-tuned", "baseline", "debugging", "modifications")),
    item("paraphrased_semantic", "600", "How can provenance metadata help address deepfakes?", "It can document where content came from and how it was changed, giving users evidence for authenticity assessments.", ("deepfake", "provenance", "authentic")),
    item("paraphrased_semantic", "600", "Why is field testing important for generative AI?", "It shows how people actually interpret, use, and act on generated information in realistic settings.", ("field testing", "interact", "make sense", "generated information")),
    item("paraphrased_semantic", "600", "How can retrieval-augmented generation reduce some output risks?", "Grounding generation in retrieved sources can improve factual support, especially when sources and citations are also reviewed and verified.", ("retrievalaugmented generation", "sources", "citations")),
    item("paraphrased_semantic", "600", "Why must content-moderation tests reflect deployment conditions?", "The effectiveness of safeguards depends on users, context, prompts, languages, and uses encountered in operation.", ("content moderation", "deployment", "testing")),
    item("paraphrased_semantic", "600", "How can generative AI threaten intellectual property?", "It may reproduce, expose, or generate material involving copyrighted, licensed, patented, proprietary, or trademarked information.", ("copyrighted", "licensed", "patented", "proprietary")),
    item("paraphrased_semantic", "600", "Why should organizations measure inference impacts separately from training impacts?", "Resource use varies by lifecycle stage and usage volume, so deployment and inference can create substantial cumulative energy or water costs.", ("inference", "training", "energy", "water")),
    item("paraphrased_semantic", "600", "How does regular public reporting support GAI governance?", "Reporting performance, feedback, and improvements increases transparency and enables stakeholders to track changes and accountability.", ("regular monitoring", "publish reports", "feedback", "improvements")),
    item("paraphrased_semantic", "600", "Why are broad benchmark scores insufficient for every GAI deployment?", "Actual risk depends on the specific task, context, population, integrations, and ways the system may be used or repurposed.", ("contexts", "repurposed", "risk mapping", "testing")),
    item("paraphrased_semantic", "600", "How can human overrides reveal weaknesses in a GAI system?", "Tracking when and why people override outputs can expose errors, provenance problems, and recurring conditions requiring improvement.", ("overrides", "content provenance", "track")),
    item("paraphrased_semantic", "600", "Why should red teams include domain expertise?", "Domain experts can recognize specialized harms, realistic misuse, and context-specific failure modes that generic testing may overlook.", ("red teaming", "domain experts", "adverse")),

    # Definitions / concepts (15)
    item("definitions_concepts", "100", "Define AI risk management.", "AI risk management is the coordinated set of activities used to direct and control an organization with regard to AI risk.", ("risk management refers", "coordinated activities", "direct and control")),
    item("definitions_concepts", "100", "What does trustworthy AI mean in the AI RMF?", "It refers to AI whose characteristics include validity, reliability, safety, security, resilience, accountability, transparency, explainability, interpretability, privacy enhancement, and managed harmful bias.", ("characteristics of trustworthy", "valid", "accountable", "privacy")),
    item("definitions_concepts", "100", "What is explainability?", "Explainability is representing the mechanisms underlying an AI system's operation in ways people can understand.", ("explainability", "mechanisms", "operation")),
    item("definitions_concepts", "100", "What is interpretability?", "Interpretability is the meaning of an AI system's output in the context of its designed functional purpose.", ("interpretability", "meaning", "output", "context")),
    item("definitions_concepts", "100", "What is an AI actor?", "An AI actor is an organization or individual that plays an active role in the AI system lifecycle.", ("ai actors", "active role", "lifecycle")),
    item("definitions_concepts", "100", "What is a residual risk?", "Residual risk is the risk remaining after risk-response measures have been applied.", ("residual", "risk tolerance", "safe")),
    item("definitions_concepts", "100", "What is a risk profile in the AI RMF?", "A profile describes how AI RMF outcomes apply to a particular use case, sector, organization, or common activity.", ("profiles", "use-case", "functions", "categories")),
    item("definitions_concepts", "100", "What is harmful bias?", "Harmful bias is bias that can produce inequitable or otherwise undesirable outcomes, including systemic, computational, and human forms.", ("harmful bias", "systemic", "computational", "human")),
    item("definitions_concepts", "600", "Define a generative AI system.", "A generative AI system uses a model to produce synthetic content such as text, images, audio, video, or other digital material.", ("generative ai", "synthetic content", "text", "images")),
    item("definitions_concepts", "600", "What is information integrity?", "Information integrity concerns the authenticity, accuracy, provenance, and reliability of information and the information ecosystem.", ("information integrity", "authenticity", "provenance")),
    item("definitions_concepts", "600", "What is prompt injection?", "Prompt injection is an attack or manipulation that uses crafted instructions to alter a GAI system's intended behavior or expose protected information.", ("prompt injection", "attack", "instructions")),
    item("definitions_concepts", "600", "What is the GAI value chain?", "It is the network of actors, components, models, data, infrastructure, and services involved in developing, deploying, and operating a GAI system.", ("value chain", "component integration", "third-party")),
    item("definitions_concepts", "600", "What is content provenance?", "Content provenance is information about the origin and modification history of digital content.", ("content provenance", "origin", "modifications")),
    item("definitions_concepts", "600", "What is AI red-teaming?", "AI red-teaming is adversarial testing intended to identify harmful behaviors, vulnerabilities, failure modes, and the limits of safeguards.", ("ai red-teaming", "adverse behavior", "safeguards")),
    item("definitions_concepts", "600", "What is harmful bias and homogenization in GAI?", "It is the risk that generated content reinforces harmful stereotypes or reduces diversity by producing overly uniform outputs and perspectives.", ("harmful bias and homogenization", "stereotypes", "diversity")),

    # Lists / numeric facts (15)
    item("lists_numeric_facts", "100", "List the seven characteristics of trustworthy AI systems highlighted by the AI RMF.", "They are valid and reliable; safe; secure and resilient; accountable and transparent; explainable and interpretable; privacy-enhanced; and fair with harmful bias managed.", ("valid and reliable", "safe", "secure and resilient", "privacy-enhanced")),
    item("lists_numeric_facts", "100", "How many top-level functions are in the AI RMF Core, and what are they?", "There are four: GOVERN, MAP, MEASURE, and MANAGE.", ("four functions", "govern", "map", "measure", "manage")),
    item("lists_numeric_facts", "100", "What four risk-response choices does MANAGE name?", "Mitigate, transfer, avoid, or accept the risk.", ("mitigating", "transferring", "avoiding", "accepting")),
    item("lists_numeric_facts", "100", "By what year did NIST expect a formal review of AI RMF 1.0?", "NIST expected a formal review no later than 2028.", ("no later than 2028", "formal input")),
    item("lists_numeric_facts", "100", "What two numbers does the AI RMF versioning system use?", "It uses a major generation number and a minor revision number, such as 1.0 and 1.1.", ("two-number versioning", "major", "minor")),
    item("lists_numeric_facts", "100", "Name the three broad sources of AI bias discussed by NIST.", "Systemic bias, computational and statistical bias, and human-cognitive bias.", ("systemic", "computational", "human", "bias")),
    item("lists_numeric_facts", "100", "List the four major groups of AI lifecycle actor tasks described in the appendix.", "Design and development; deployment; operation and monitoring; and test, evaluation, verification, and validation.", ("design and development", "deployment", "operation and monitoring", "test", "validation")),
    item("lists_numeric_facts", "100", "What three kinds of methods may MEASURE use?", "Quantitative, qualitative, or mixed-method approaches.", ("quantitative", "qualitative", "mixed")),
    item("lists_numeric_facts", "600", "How many primary GAI risk areas are listed in the profile?", "The profile identifies 12 primary generative AI risk areas.", ("12", "gai risks", "confabulation")),
    item("lists_numeric_facts", "600", "Name four types of content a generative AI system may produce.", "Examples include text, images, audio, and video.", ("text", "image", "audio", "video", "generative")),
    item("lists_numeric_facts", "600", "List three environmental resources or effects that GAI evaluation may track.", "Examples include energy use, water consumption, and other environmental impacts associated with training, fine-tuning, and inference.", ("energy", "water consumption", "environmental impacts")),
    item("lists_numeric_facts", "600", "Which three groups should be involved in GAI system evaluations under MP-3.4-006?", "End users, practitioners, and operators.", ("mp-3.4-006", "end-users", "practitioners", "operators")),
    item("lists_numeric_facts", "600", "List four categories of protected or controlled information that GAI can expose.", "Examples include copyrighted, licensed, patented, proprietary, personal, sensitive, or trade-secret information.", ("copyrighted", "licensed", "patented", "proprietary")),
    item("lists_numeric_facts", "600", "What three items should regular GAI monitoring reports describe?", "System performance, feedback received, and improvements made.", ("performance", "feedback received", "improvements made")),
    item("lists_numeric_facts", "600", "Name three approaches for learning how people respond to generated content.", "Focus groups, small user studies, and surveys are examples; field testing is another approach.", ("focus groups", "small user studies", "surveys")),

    # Comparison / reasoning (10)
    item("comparison_reasoning", "100", "How do explainability and interpretability differ?", "Explainability concerns how the system's mechanisms operate, while interpretability concerns what its output means in context.", ("explainability", "interpretability", "mechanisms", "meaning")),
    item("comparison_reasoning", "100", "How do validity and reliability differ from accuracy?", "Validity and reliability concern whether requirements are met consistently under expected conditions, while accuracy is closeness to accepted true values.", ("validity", "reliability", "accuracy", "true values")),
    item("comparison_reasoning", "100", "How are accountability and transparency related but distinct?", "Transparency supplies appropriate information about a system, while accountability assigns responsibility for decisions, outcomes, and risk management.", ("accountability", "transparency", "responsibility")),
    item("comparison_reasoning", "100", "Why are AI risks harder to test than many traditional software risks?", "AI behavior depends heavily on data and context, can drift, has emergent and probabilistic failure modes, and often lacks fully specified expected outputs.", ("traditional software", "testing", "data", "emergent")),
    item("comparison_reasoning", "100", "What is the difference between an AI system operator and an affected individual?", "An operator runs or monitors the system, whereas an affected individual may experience its consequences without using or interacting with it.", ("system operators", "affected individuals", "interact")),
    item("comparison_reasoning", "600", "How do pre-deployment testing and post-deployment monitoring complement each other?", "Pre-deployment testing finds anticipated failures before release; monitoring detects real-world changes, incidents, feedback, and emergent risks after release.", ("pre-deployment", "post-deployment", "monitoring")),
    item("comparison_reasoning", "600", "How is confabulation different from deliberate misinformation?", "Confabulation is erroneous generated content that need not be intentionally deceptive; misinformation or disinformation concerns false content and, for disinformation, deliberate intent to deceive.", ("confabulation", "misinformation", "false")),
    item("comparison_reasoning", "600", "How do watermarking and provenance records address synthetic content differently?", "Watermarking embeds or associates a detectable signal, while provenance records document origin and modification history across the content lifecycle.", ("watermark", "provenance", "origin")),
    item("comparison_reasoning", "600", "Why should a fine-tuned model be compared with its baseline?", "Comparison isolates changes caused by fine-tuning or added data and helps diagnose whether adaptations introduced new failures or risks.", ("fine-tuning", "baseline", "debugging")),
    item("comparison_reasoning", "600", "How do laboratory user studies and field testing provide different evidence?", "Controlled studies isolate specific interactions, while field testing reveals behavior and impacts in realistic contexts of use.", ("user studies", "field testing", "interact")),

    # Multi-chunk (10)
    item("multi_chunk", "100", "How do the MAP and MEASURE functions contribute differently to AI risk management?", "MAP establishes context and identifies risks, while MEASURE analyzes, evaluates, and monitors those risks.", ("map 1", "context is established", "map function"), ("measure function employs", "quantitative", "qualitative", "analyze")),
    item("multi_chunk", "100", "How do GOVERN and MANAGE work together across the AI risk lifecycle?", "GOVERN establishes culture, policies, roles, and oversight; MANAGE prioritizes and treats identified risks using those organizational structures.", ("govern function", "culture", "policies"), ("manage", "prioritized", "risk response")),
    item("multi_chunk", "100", "How do MAP, MEASURE, and MANAGE form a continuous risk-management loop?", "MAP frames context and risks, MEASURE assesses them, and MANAGE prioritizes and responds; monitoring results then inform continued mapping and measurement.", ("map 1", "context is established", "map function"), ("measure function employs", "quantitative", "monitor"), ("manage function entails", "prioritized", "respond")),
    item("multi_chunk", "100", "How do privacy and fairness considerations interact in trustworthy AI?", "Privacy controls protect personal data, while fairness work identifies and manages harmful bias; both must be balanced with other trustworthiness characteristics in context.", ("privacy-enhanced", "de-identification"), ("fair", "harmful bias")),
    item("multi_chunk", "100", "Compare the responsibilities of AI designers, operators, and TEVV actors.", "Designers define concepts and build systems, operators run and monitor them, and TEVV actors test, evaluate, verify, and validate throughout the lifecycle.", ("ai design actors", "planning", "design"), ("operation and monitoring", "system operators"), ("tevv", "verification", "validation")),
    item("multi_chunk", "600", "How can provenance controls and human review jointly reduce misleading GAI output?", "Provenance controls expose origin and modification history, while human reviewers verify claims, sources, and citations before relying on outputs.", ("trace the origin", "modifications", "digital content"), ("review and verify", "sources", "citations")),
    item("multi_chunk", "600", "How should organizations assess and manage the environmental impact of GAI?", "They should measure energy and water use across training, fine-tuning, inference, and deployment, then use those results in governance and risk-management decisions.", ("environmental impacts", "energy", "water consumption"), ("training", "fine tuning", "deploying models")),
    item("multi_chunk", "600", "How do red-teaming, field testing, and ongoing monitoring cover different GAI risks?", "Red-teaming probes adversarial failures, field testing studies real human use, and ongoing monitoring detects operational changes, feedback, and incidents after deployment.", ("red-teaming", "stress test", "safeguards"), ("field testing", "people interact"), ("regular monitoring", "feedback", "improvements")),
    item("multi_chunk", "600", "How can third-party model governance address changes introduced by fine-tuning?", "Organizations should document suppliers and components, retain or compare baselines, reassess adapted models, and apply their own risk tolerance after fine-tuning.", ("third-party models", "fine-tuned", "re-evaluate"), ("baseline", "debugging", "modifications"), ("risk tolerance", "reassess risk")),
    item("multi_chunk", "600", "How do confabulation and information-integrity risks reinforce each other?", "Confabulated outputs introduce plausible falsehoods, and at scale those outputs can pollute the information ecosystem; verification and provenance help limit both risks.", ("confabulation", "false", "confident"), ("information integrity", "provenance", "authenticity")),
]


def score(text, terms):
    normalized = re.sub(r"\s+", " ", text.lower())
    total = 0
    for term in terms:
        term = term.lower()
        if term in normalized:
            total += 20 + len(term.split()) * 5 + normalized.count(term)
        else:
            total += sum(1 for word in re.findall(r"[a-z0-9]+", term) if word in normalized)
    return total


def best_chunks(chunks, doc_key, groups):
    candidates = [c for c in chunks if f"-{doc_key}-" in c["chunk_id"]]
    selected = []
    for terms in groups:
        ranked = sorted(candidates, key=lambda c: (score(c["text"], terms), -c["chunk_number"]), reverse=True)
        choice = next((c for c in ranked if c["chunk_id"] not in selected), ranked[0])
        selected.append(choice["chunk_id"])
    return selected


def main():
    chunks_by_size = {}
    for size in (256, 512):
        chunks_by_size[size] = json.loads((PROCESSED / f"chunks_{size}.json").read_text(encoding="utf-8"))

    records = []
    for number, spec in enumerate(SPECS, start=1):
        doc_id = "NIST.AI.100-1.pdf" if spec["document_key"] == "100" else "NIST.AI.600-1.pdf"
        records.append({
            "query_id": f"q{number:03d}",
            "query": spec["query"],
            "query_type": spec["query_type"],
            "document_id": doc_id,
            "expected_answer": spec["expected_answer"],
            "relevant_chunks_256": best_chunks(chunks_by_size[256], spec["document_key"], spec["anchor_groups"]),
            "relevant_chunks_512": best_chunks(chunks_by_size[512], spec["document_key"], spec["anchor_groups"]),
        })

    expected = {
        "direct_lexical": 20,
        "paraphrased_semantic": 30,
        "definitions_concepts": 15,
        "lists_numeric_facts": 15,
        "comparison_reasoning": 10,
        "multi_chunk": 10,
    }
    counts = Counter(record["query_type"] for record in records)
    assert len(records) == 100, len(records)
    assert counts == Counter(expected), counts

    for size in (256, 512):
        valid_ids = {chunk["chunk_id"] for chunk in chunks_by_size[size]}
        for record in records:
            ids = record[f"relevant_chunks_{size}"]
            assert ids and all(chunk_id in valid_ids for chunk_id in ids)
            expected_refs = 1 if record["query_type"] != "multi_chunk" else len(SPECS[int(record["query_id"][1:]) - 1]["anchor_groups"])
            assert len(ids) == expected_refs

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} queries to {OUTPUT}")
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
