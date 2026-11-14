## Output token budgets (2026-11-08)

The operator requested larger allowances after the November 8 Claude forecast used
all 64,000 completion tokens as reasoning and returned no answer. Production OpenAI
and Anthropic forecasters, stackers and utility/parser paths now allow 128,000 tokens
(shared reasoning plus visible answer). The Google forecaster allows 65,536.

Verified against the live [OpenRouter model list](https://openrouter.ai/api/v1/models)
and per-model `/endpoints` metadata on November 8: Sol 6.1, Terra 6.1, Luna 6 and
Opus 5.5 advertise 128,000. Argon's headline is 943,718, but its endpoint list also
contains 65,536-token routes; the smaller ceiling preserves routing compatibility.
Do not copy the headline maximum blindly across routes.

Gemini 4 Carbon replaced Argon in the Google slot on 2026-11-13 and kept the 65,536 cap; Argon was
restored on 2026-11-14 after Carbon timed out (see roster_history.md). Its
`/endpoints` list read that day shows seven routes: four at 943,718, the Vertex US and EU routes at
65,536, and the plain Google AI Studio route at 32,768. A 65,536 request may skip that one route
rather than fit it; dropping to 32,768 to keep it would leave little room for an answer at `high`
effort, which is the failure the November 8 change was made for.

Omitting `max_tokens` is not unlimited: [OpenRouter omits it upstream](https://openrouter.ai/docs/api_reference/parameters),
leaving provider defaults. Those defaults need not be the advertised maximum, so
we use explicit verified ceilings instead. Remaining context space still applies.
No model roster, reasoning effort, retry behavior, deadline, or degradation alert
changed. Larger allowances can increase spend and latency when calls exceed the old
cap; unchanged wall deadlines can still interrupt long generations. Paid validation
was not run. This is a configuration-era change when merged, not a new model roster.

Native research, agentic tool loops and offline/legacy benchmark-specific caps are
separate task budgets and are unchanged; native Gemini research already omits its
output cap. This change does not fix its upstream 504 timeout.
