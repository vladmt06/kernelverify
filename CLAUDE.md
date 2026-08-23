# kernelverify

Repository-specific instructions live in AGENTS.md. Read it before doing
anything substantial here; this file only routes skills.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill
tool. When in doubt, invoke the skill.

MANDATORY: whenever planning anything or assigning a task to any agent, use
the superpowers skills. Planning goes through superpowers:brainstorming then
superpowers:writing-plans; task assignment and dispatch go through
superpowers:subagent-driven-development (or superpowers:executing-plans for
inline execution). This applies every time, with no exceptions.

Key routing rules:
- Planning any work or writing any plan → superpowers:brainstorming then superpowers:writing-plans
- Assigning or dispatching a task to any agent → superpowers:subagent-driven-development
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate (with superpowers:systematic-debugging)
- QA/testing behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
