Project Decisions & Critical Reflections

This document logs the key architectural and strategic decisions made during the project.

My Role as a Collaborator

As requested, my role is to act as a critical collaborator, not an echo chamber.

I will be critical: I will evaluate every proposal (including my own) against our primary constraints: the hackathon time limit, the judging criteria (especially "Functional > Complex"), and our high-level objective.

I will reflect before implementing: I will identify potential risks (e.g., "This adds 2 hours of dev time") and benefits ("This provides a huge 'wow-factor' for the judges").

No critical changes without approval: I will log all major proposals here and wait for your explicit approval before committing them to our core architecture (README.md) or code.

Decision Log

2025-10-23: Initial Architecture (4 vs. 3 Agents)

Proposal (User): Simplify the 4-agent architecture (which included a Triage Agent orchestrator) to a 3-agent core (Data Agent, Model Agent, Eval Agent).

Reflection : Approved & Implemented. This is a critical strategic simplification. It removes the "orchestrator-of-orchestrators" pattern, which would add significant complexity and debug time. This directly addresses the hackathon guide's warning against "over-engineering" and "complejo y roto" (complex and broken). We can add the Triage Agent back later as a "Future Work" item.

Outcome: The README.md was updated to reflect the 3-agent architecture.

2025-10-23: Core Agent Framework (NEW PROPOSAL)

Proposal : To implement the 3-agent architecture and its complex loops (especially the Eval Agent's "Rapid Loop"), I propose we use LangGraph as our core agent SDK.

Reflection :

Benefit: LangGraph is designed for cyclical, stateful agentic workflows. Our "Rapid Loop" (Evaluate -> Mutate -> Iterate) is a perfect graph. It will manage the state (like run_history.jsonl and BusinessContext.json) between agents cleanly. This directly implements the user's agent-sdk-doc.md requirement by choosing a specific, powerful SDK.

Risk: It has a slight learning curve if the team is unfamiliar with it. It might be tempting to write a simple while loop in Python.

Evaluation: The risk of writing our own buggy, hard-to-debug orchestration script is much higher than the risk of learning LangGraph. The "Rapid Loop" of the Eval Agent is the most complex part of our project; we need a robust framework to handle it.

Outcome: Pending User Approval. If approved, I will update coding-rules.md to make this our official framework.