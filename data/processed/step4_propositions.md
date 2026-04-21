# Step 4 -- Strategic Linkage: Propositions

**Paper**: Strategic AI Orientation Enabled by Trust & Integration Readiness
**Venue**: AMCIS 2026

## Conceptual Model

```
Upper Echelons / Strategic Choice Layer
  CIO Centrality + Board AI Awareness
           |
           v
     AI ORIENTATION  <---- Regulatory Pressure (P5 moderator)
    (strategic intent
     toward AI)      
           |
    ______/ \______
   |                |
   v                v
TRUST           INTEGRATION
READINESS       READINESS
(governance     (architecture
 capability)     capability)
   |                |
   |_____ x ________|   <-- Complementarity (P4)
           |
           v
   REALIZED AI VALUE
   (production scale,
    safe deployment,
    competitive advantage)
```

---

## P1: CIO Centrality → AI Orientation

**Proposition P1**: The greater the CIO's centrality in the firm's strategic decision-making (as reflected in reporting structure, board access, and AI agenda ownership), the stronger the firm's AI orientation.

**Role in Model**: Antecedent (independent variable)

**Mechanism**: CIO attention allocation and strategic agenda-setting

### Rationale

Upper echelons theory (Hambrick & Mason, 1984) holds that executive cognition, values, and attention shape strategic choices. CIOs who sit at the strategic apex — with direct board access and ownership of the AI agenda — translate their technical and organizational understanding of AI into firm-level strategic commitment. BCG (2025) finds that at 86% of AI future-built companies, IT leads or co-leads GenAI initiatives (vs. 54% for AI-stagnating firms), providing empirical support for the CIO centrality → orientation link. McKinsey (2026) frames the CIO mandate as explicitly strategic: 'strategy, speed, and scaled intelligence' — not merely operational delivery. Gartner (2026) identifies agility and risk mastery as the defining CIO agenda items, both requiring strategic-level AI orientation.

### Grounding Sources

- MISQ Vol.45 No.3 -- Strategic Directions for AI: CIOs and Boards [#02 -- PENDING, primary anchor]
- BCG (2025) -- CIO's Role in AI Value Creation [#20]: 86% vs 54% IT leadership statistic
- McKinsey (2026) -- New CIO Mandate [#32]: strategy, speed, scaled intelligence framing
- Gartner (2026) -- CIO Agenda 2026 [#07]: agility and risk as CIO priorities
- Springer (2023) -- CIO Role in Digital Transformation [#01]: CIO multidimensionality in DT
- SAGE/Hossain et al. (2025) -- Digital Leadership Dynamic Managerial Capability [#03]

### Evidence from Step 2 (Crosswalk)

N/A -- P1 concerns strategic intent, not operational controls.

### Evidence from Step 3 (ATLAS Incident Validation)

N/A -- P1 is an antecedent proposition, not directly tested by incident data.

### Falsifiability

P1 would be falsified if firms with peripheral CIOs (low board access, IT-only mandate) demonstrate equivalent AI orientation to firms with strategically central CIOs. Testable via CIO reporting structure data + AI orientation proxy (10-K AI language, AI investment as % of IT budget).

---

## P2: Trust Readiness mediates/moderates AI Orientation → Realized Value

**Proposition P2**: Higher trust readiness (governance capability bundle) strengthens the conversion of AI orientation into production-scale AI value by reducing incident risk, enabling auditability, and satisfying regulatory license-to-operate requirements.

**Role in Model**: Mediator/Moderator (enabler of orientation → value conversion)

**Mechanism**: Trust readiness reduces the probability and severity of AI incidents (direct risk reduction), satisfies regulatory license-to-operate requirements (compliance enablement), and builds stakeholder confidence enabling broader deployment authorization (organizational legitimacy).

### Rationale

AI orientation without trust readiness creates a capability gap: firms with strong strategic intent but weak governance controls cannot safely scale AI to production. NIST AI RMF (2023) frames governance, mapping, measuring, and managing as the preconditions for trustworthy AI deployment — not optional add-ons. The EU AI Act (Art. 9-15) makes trust readiness legally mandatory for high-risk AI, creating a hard regulatory constraint on orientation realization. MITRE ATLAS incident analysis (n=52) confirms the cost of trust readiness gaps: Impact (47x), Defense Evasion (34x), and Model Access (24x) are the dominant failure modes — all preventable through TR-3 (monitoring), TR-2 (threat modeling), and TR-1 (access governance). The most-needed missing control across incidents is AI Telemetry Logging (33 incidents) — a basic trust readiness capability. BCG (2025) finds that 74% of AI future-built companies continuously monitor RAI framework compliance, vs. far fewer among laggards — directly linking trust readiness maturity to AI value realization outcomes.

### Grounding Sources

- NIST AI RMF 1.0 [#21]: GOVERN/MAP/MEASURE/MANAGE functions as preconditions
- NIST GenAI Profile [#22]: agentic-specific risk actions
- EU AI Act 2024/1689 [#26]: Art. 9-15 mandatory governance obligations
- MITRE ATLAS [atlas-data]: 52 incidents, trust gaps as dominant failure pattern
- OWASP Top 10 LLM [#29]: 10 operationalized trust threats
- BCG (2025) [#20]: 74% of AI future-built firms monitor RAI compliance
- ISACA COBIT for AI [#25]: governance objectives for AI systems
- McKinsey -- Agentic AI Safety Playbook [#15]: safety/security as scaling prerequisite

### Evidence from Step 2 (Crosswalk)

Step 2 crosswalk: 31 of 42 competency statements map to Trust Readiness bundle. Evaluation & Monitoring Infrastructure addresses 12 governance requirements — the single most cross-cutting control in the framework.

### Evidence from Step 3 (ATLAS Incident Validation)

52 ATLAS case studies show that 100% of incidents involved at least one missing trust readiness control. Top missing controls: AI Telemetry Logging (33x), Adversarial Input Detection (25x), Model Hardening (23x), Access Controls (20x). These map directly to TR-3, TR-2, TR-1, and TR-4 sub-competencies.

### Falsifiability

P2 would be falsified if firms with low trust readiness scores achieve equivalent AI value realization to high-trust-readiness firms with similar orientation levels. Testable via: trust readiness operationalized as NIST AI RMF maturity score or RAI program presence; value realization via AI revenue contribution or AI-at-scale deployment count.

---

## P3: Integration Readiness mediates/moderates AI Orientation → Realized Value

**Proposition P3**: Higher integration readiness (architecture capability bundle) strengthens the conversion of AI orientation into production-scale AI value by enabling reusable agentic architectures, operational AI lifecycle management, and scalable deployment without accumulating prohibitive technical debt.

**Role in Model**: Mediator/Moderator (enabler of orientation → value conversion)

**Mechanism**: Integration readiness reduces the time-to-production for AI use cases (velocity), enables reuse across use cases via archetypes (efficiency), prevents shadow IT proliferation via governed architecture (control), and ensures operational sustainability via GenAIOps practices (resilience).

### Rationale

BCG (2025) finds that more than 50% of firms report legacy IT architectures as the primary barrier to AI scaling — directly implicating integration readiness as the binding constraint between orientation and value. Deloitte Tech Trends (2026) explicitly argues that agentic AI value comes from redesigning operations and building agent-compatible architectures, not from model selection alone. McKinsey (2026) identifies scalability, reusability, and data availability as the three key challenges CIOs must solve to convert GenAI orientation into realized value. Microsoft Azure Well-Architected AI addresses nondeterminism, data/app design, and GenAIOps as the core operational concerns for AI workloads. MITRE ATLAS confirms the integration dimension: Resource Development (AML.TA0003, 77x) and Initial Access (AML.TA0004, 41x) failures reflect weak supply chain and access boundary architecture — both integration readiness failures.

### Grounding Sources

- BCG (2025) [#20]: 50%+ cite legacy architecture as AI scaling barrier
- Deloitte Tech Trends 2026 [#09]: agent-compatible architectures as value source
- McKinsey (2026) New CIO Mandate [#32]: scalability, reusability, data availability
- Microsoft Learn LLMOps [#11]: GenAIOps as operational management discipline
- Bizzdesign [#10]: enterprise architecture facing AI-era integration demands
- McKinsey Agentic AI Safety Playbook [#15]: architecture as safety prerequisite
- MITRE ATLAS [atlas-data]: Resource Development (77x) and Initial Access (41x) failures

### Evidence from Step 2 (Crosswalk)

Step 2 crosswalk: GenAIOps/MLOps Lifecycle Governance addresses 10 governance requirements — second most cross-cutting control. Tool-Use Boundaries addresses 8 requirements spanning NIST, EU AI Act, and OWASP sources.

### Evidence from Step 3 (ATLAS Incident Validation)

ATLAS tactic frequency: Resource Development (AML.TA0003, 77x) -- supply chain and resource acquisition failures map to IR-6 (modular architecture) and IR-4 (RAG/data grounding). Initial Access (AML.TA0004, 41x) maps to IR-2 (tool-use boundaries). Execution (AML.TA0005, 37x) maps to IR-1 (orchestration pattern controls) and IR-3 (nondeterminism management).

### Falsifiability

P3 would be falsified if firms with low integration readiness (high technical debt, no GenAIOps practices, monolithic architectures) achieve equivalent AI scaling outcomes to high-integration-readiness firms. Testable via: integration readiness operationalized as GenAIOps maturity, architecture modularity score, or AI use case reuse rate; value via production AI deployment count and time-to-scale.

---

## P4: Trust × Integration Readiness Complementarity

**Proposition P4**: Trust readiness and integration readiness are strategic complements: the marginal value of each capability bundle is higher when the other bundle is also present, such that weak architecture undermines governance effectiveness and weak governance constrains safe architecture deployment.

**Role in Model**: Interaction effect (joint moderator of orientation → value)

**Mechanism**: Architectural controls provide the technical substrate that makes governance policies enforceable and measurable. Governance policies provide the authorization and boundary conditions within which architectural patterns can be safely deployed at scale. Each bundle amplifies the other's marginal return.

### Rationale

Complementarity exists when capabilities reinforce each other's effectiveness (Milgrom & Roberts, 1990). Trust and integration readiness are complementary because: (1) governance controls require architectural hooks to be enforceable (e.g., audit logging requires telemetry infrastructure; human override requires architectural approval gates); (2) architectural patterns require governance boundaries to be safe (e.g., multi-agent orchestration without access governance creates OWASP LLM06 excessive agency risk; RAG without data governance creates LLM08 embedding vulnerabilities). NIST AI RMF explicitly treats GOVERN, MAP, MEASURE, and MANAGE as interdependent functions — governance without measurement infrastructure is unverifiable; measurement without governance is ungoverned. The EU AI Act (Art. 9-15) similarly requires both policy documentation (trust) and technical implementation (integration) as co-equal obligations. MITRE ATLAS case studies show that incidents typically involve failures in both domains simultaneously: e.g., a supply chain attack (integration: weak vendor controls) succeeds because monitoring is absent (trust: no telemetry logging).

### Grounding Sources

- NIST AI RMF 1.0 [#21]: GOVERN/MAP/MEASURE/MANAGE interdependence
- EU AI Act 2024/1689 [#26]: co-equal policy + technical obligations (Art. 9-15)
- OWASP Top 10 LLM [#29]: LLM06 (governance) requires IR-2 (architecture)
- MITRE ATLAS [atlas-data]: co-occurring trust + integration failures in incidents
- McKinsey Agentic AI Safety Playbook [#15]: safety requires both governance and architecture
- BCG (2025) [#20]: governance framework + modular architecture as joint CIO mandate

### Evidence from Step 2 (Crosswalk)

Step 2 crosswalk: 8 of 18 architecture controls appear in both Trust and Integration Readiness governance requirement rows, confirming structural interdependence. Human-in-the-Loop Approval Gates, for example, addresses 8 governance requirements spanning both NIST GOVERN and EU Art. 14.

### Evidence from Step 3 (ATLAS Incident Validation)

Of 52 ATLAS case studies, the dominant failure pattern combines integration weaknesses (Resource Development 77x, Initial Access 41x) with trust gaps (missing telemetry 33x, missing access controls 20x) — consistent with co-occurring failures across both bundles rather than isolated single-bundle gaps.

### Falsifiability

P4 would be falsified if trust readiness and integration readiness independently predict AI value realization with no significant interaction term. Testable via regression with interaction term (TR × IR) on AI value outcome; complementarity confirmed if interaction coefficient is positive and significant beyond main effects.

---

## P5: Regulatory Pressure → Trust Readiness Salience → AI Orientation Agenda

**Proposition P5**: External regulatory pressure (operationalized as EU AI Act applicability or equivalent jurisdiction-level AI regulation) increases the strategic salience of trust readiness, elevating governance capability-building onto the CIO and board AI orientation agenda as a license-to-operate condition.

**Role in Model**: Boundary condition / contextual moderator of trust readiness salience

**Mechanism**: Regulatory mandates create coercive isomorphic pressure that elevates trust readiness from operational compliance to strategic priority. CIOs and boards respond by incorporating governance capability-building into AI orientation decisions — effectively making trust readiness a prerequisite for ambitious AI orientation rather than a downstream consequence of it.

### Rationale

Institutional theory (DiMaggio & Powell, 1983) predicts that coercive isomorphism — regulatory mandates — forces organizations to adopt governance structures regardless of internal strategic preference. The EU AI Act (Regulation 2024/1689) creates binding obligations for high-risk AI systems (Art. 9-15) and GPAI models with systemic risk (Art. 53-55), making trust readiness a legal prerequisite for AI deployment in EU markets. BCG (2025) identifies regulatory compliance as one of three key challenges CIOs must own to enable AI transformation. McKinsey (2026) notes that 31 US states had passed AI legislation by August 2024, with more expected — extending regulatory pressure beyond the EU. The EU Parliament briefing (2025) frames AI Act implementation as an active governance challenge requiring organizational capability investment. This regulatory pressure mechanism converts trust readiness from an optional governance best practice into a strategic imperative that shapes the AI orientation agenda itself — making governance capability a board-level concern, not just an IT compliance task.

### Grounding Sources

- EU AI Act 2024/1689 [#26]: binding obligations Art. 9-15, 53-55
- EU Parliament AI Implementation Briefing 2025 [#27]: implementation as active challenge
- EDPS GenAI Data Protection Guidance 2025 [#28]: data protection as governance constraint
- BCG (2025) [#20]: regulatory compliance as one of three CIO AI transformation challenges
- McKinsey (2026) New CIO Mandate [#32]: 31 US states with AI legislation by Aug 2024
- ISACA COBIT for AI [#25]: governance objectives connecting AI to regulatory compliance
- Gartner (2026) CIO Agenda [#07]: risk mastery as defining CIO priority

### Evidence from Step 2 (Crosswalk)

Step 2 crosswalk: EU AI Act contributes 9 of 42 governance requirements. Regulatory Compliance Documentation is demanded by 7 governance requirements across NIST, EU AI Act, and OWASP sources — confirming cross-source regulatory salience. EU Art. 9 (lifecycle risk management) alone maps to 3 architecture controls: GenAIOps, AI Risk Policy, and Regulatory Compliance Documentation.

### Evidence from Step 3 (ATLAS Incident Validation)

ATLAS mitigations in the Policy category (AML.M0005 Control Access, AML.M0019 Production Access Controls) appear in 20x and 17x incidents respectively — confirming that policy-level controls (the domain of regulatory compliance) are among the most frequently missing. Regulatory pressure that mandates these controls would directly address the most common incident root causes.

### Falsifiability

P5 would be falsified if firms subject to strong AI regulation (EU AI Act high-risk classification) show no greater trust readiness investment or board-level governance attention than unregulated peers with equivalent AI orientation. Testable via regulatory exposure as moderator in trust readiness investment models, using EU AI Act risk classification as instrument.

---

## Integration with Steps 1-3

| Step | Output | Role in Propositions |
|------|--------|----------------------|
| Step 1 | 3 construct definitions, 16 sub-competencies | Provides the theoretical vocabulary for P1-P5 |
| Step 2 | 42 requirements x 18 controls crosswalk | Grounds P2/P3/P4 in specific governance-architecture linkages |
| Step 3 | 52 ATLAS incidents coded | Validates P2/P3/P4/P5 via real-world failure mode evidence |
| Step 4 | 5 propositions | Ties constructs to IS theory for AMCIS contribution |

## Pending: MISQ Paper (#02)

Once obtained, the MISQ paper (Strategic Directions for AI: Role of CIOs and Boards) 
will strengthen P1 and P2 by providing:
- Empirical evidence for CIO/board effects on AI orientation (direct grounding for P1)
- Construct validity for AI orientation as an IS-theory construct
- Upper echelons theory operationalization in an AI context

All five propositions are currently grounded in practitioner evidence (BCG, McKinsey, 
Gartner, Deloitte) and normative standards (NIST, EU AI Act, OWASP, ATLAS). 
The MISQ paper will add peer-reviewed IS-theory legitimacy to P1 specifically.