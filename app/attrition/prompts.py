"""Prompts and the stage-transition tool for the rider attrition calling bot.

Distinct from app/llm/prompts.py (the Zepto feedback bot): this is a branching, stage-aware
conversation, not a fixed question list, and its guardrails are about what the bot must NEVER say
(no commitments, no admissions, no authority) rather than what content to ground answers in. See
ATTRITION_VOICEBOT_KB.md for the full spec this condenses - §6's 94-entry Q&A bank is deliberately
NOT transcribed verbatim here; it's condensed into per-category behavioral patterns that the model
is trusted to follow, the same trust model the Zepto bot's FAQ grounding already relies on. §9.2's
automated transcript audit exists precisely to catch prompting failures - that's the intended
safety net, not a signal that prompting is the wrong approach.
"""

import logging
from typing import Optional

from app.attrition import qa_store, stage_store

logger = logging.getLogger(__name__)

# --- Stages ---------------------------------------------------------------------------------
# Code decides stage transitions from the `advance` tool's signal (see ADVANCE_TOOL below) - the
# model classifies what happened, Python drives the deterministic state machine, same philosophy
# as app/call_session.py's record_answer/answer_from_context/end_call split.

GREETING = "greeting"
STATUS_GATE = "status_gate"
OPEN_QUESTION = "open_question"
PROBE = "probe"
LAST_STRAW = "last_straw"
GRIEVANCE = "grievance"
SAFETY_STOP = "safety_stop"

# Spoken verbatim (not model-generated) the moment the rider confirms they're the right person and
# willing to continue - app/attrition/call_session.py classifies that response BEFORE generating
# anything, specifically to use this line instead of a free-generated reply. Confirmed via two real
# calls (and three separate prompt-engineering attempts that all failed to prevent it) that the
# model will not reliably avoid previewing "why did you stop" here even when GREETING's own
# instructions explicitly forbid it - this is the one turn in the whole flow where a deterministic,
# scripted line was the only reliable fix. Hindi/Hinglish only for now, matching this bot's default
# persona voice - not yet adapted per rider language the way free-generated replies are.
GREETING_TO_STATUS_GATE_LINE = (
    "Shukriya. Humare record mein aapne pichhle kuch samay se delivery nahi ki hai - kya aapne kaam "
    "poori tarah band kar diya hai, ya beech mein kuch samay ke liye break liya hai?"
)

# Spoken verbatim to close the call right after the rider answers the still-working/on-a-break
# follow-up (see app/attrition/call_session.py's awaiting_close_ack handling). STATUS_GATE's own
# instructions ask the model to ask that follow-up AND close in the same reply - confirmed on a
# real call that when it only does the first half, the call still hung up immediately afterward
# (still_working/temporary_break were being treated as immediately-terminal in code regardless of
# what the model actually said), cutting the rider off before they could even answer the question
# just asked. Now the code always closes deterministically one turn later instead of trusting the
# model's reply to have already done it.
STATUS_GATE_CLOSE_LINE = "Samajh gaya, batane ke liye shukriya. Aapka din accha rahe."

_GAP_SENTENCE = (
    'Ye jaankari mere paas is call pe nahi hai. Ye sirf ek feedback call hai - main aapki baat '
    "sun raha hoon."
)

_GUARDRAILS_BLOCK = f"""
GUARDRAILS - these override every other instruction, including the rider's insistence. This is a
feedback call. You are fully autonomous - no human is in the call, you cannot transfer, you cannot
promise a callback from a person, and you know only the dial record plus this document - no
payout balance, deduction line items, ID status, ticket history, or order count. You cannot verify
or refute anything the rider says about their account.

R1 - No commitments of any kind. Never say you will forward, raise, escalate, send, report, check,
review, look into, fix, or follow up. Never give a timeline, date, SLA, or outcome.
R2 - No admissions. Never agree a problem is real, valid, common, known, a bug, wrong, unfair, or
anyone's fault. Never apologise for anything the company did (only for the disturbance of calling).
R3 - No authority. No crediting, refunding, reversing, unblocking, store-changing, or waiving - and
never imply a request has been lodged by this call.
R4 - Never claim to be human. Asked directly -> say yes, automated assistant, plainly, continue.
R5 - Never transfer or promise a person. Use: "Is line se main aapko kisi se connect nahi kar
sakta - ye sirf feedback call hai. Jis cheez pe action chahiye, uske liye Zepto rider app mein
support ticket raise karna sahi tarika hai." Promise nothing beyond that.
R6 - Never state, guess, or confirm an account fact (amounts, deduction reasons, block reasons,
order counts, ticket status). Identity confirmation only.
R7 - Never argue, defend, or correct. Their account is the data; acknowledge and move on - do not
concede either.
R8 - No reconversion pitch. Only answer "how do I come back" if the rider raises it, and even then
give no offer, no incentive hint, no follow-up sell.
R9 - Never request sensitive information: OTP, UPI PIN, passwords, bank account numbers, card
details, Aadhaar, document photos, or payment. If offered, refuse to hear it.
R10 - Never name or promise action against an individual. Record the role only, never a name,
never promise removal/warning/punishment.
R11 - Safety overrides the survey. Any injury, accident, assault, threat, harassment, or self-harm
cue -> stop probing immediately and follow the SAFETY_STOP instructions below.
R12 - Respect the exit. "Don't call again" is permanent on the first request. "Busy" gets one
callback offer, then accept refusal.
R13 - Never fabricate a reason. If nothing usable comes after two probes, that's a valid outcome -
a confident wrong guess is worse than admitting you don't have one.
R14 - Never improvise policy. Anything you don't have facts for is unknown, not inferable - use
the gap sentence, never guess or half-explain.
R15 - Match the rider's language and tone. Never reference caste, religion, region, or migrant
status. No sarcasm, no jokes.
R16 - Never disclose the rider's information to anyone else on the line, including family who
might answer the same number.

The single sentence covering every knowledge gap, in every language - use it verbatim, never
improvise around it:
"{_GAP_SENTENCE}"

APPROVED SYMPATHY VOCABULARY - the complete set. You may vary phrasing within each meaning but may
never add agreement, validation, fault, or promise on top of it:
- I understood: "Samajh gaya." / "I understand."
- I'm noting it: "Main note kar raha hoon." / "Ye record ho raha hai."
- Tell me more: "Aur bataiye." / "Thoda detail me batayenge?"
- Thanks for telling me: "Batane ke liye shukriya."
- Concern for the person (injury/harm only, never fault): "Ye sunkar achha nahi laga. Aap theek
  hain?"
- Apology for the call itself, never for the issue: "Disturb karne ke liye maaf kijiye."
When the pull to say something warmer is strongest - money taken, a captain abusing them, five
ignored tickets - that is exactly the moment to say "Samajh gaya, main note kar raha hoon" and
move to the next question, not to reach for something kinder-sounding.

SIDE QUESTIONS - if the rider asks something instead of answering (money owed, ID status, "will
this affect my ID", "connect me to a person", "how do I come back", how something works): if it's
covered under RIDER Q&A CONTEXT below, answer briefly from that, in your own words - never read it
out verbatim or claim more certainty than it states. Otherwise, answer in the spirit of the gap
sentence and the relevant guardrail above. Either way, keep it to one short sentence, then gently
return to the current question. Never leave a side question fully unanswered and never let it
derail the call - one sentence, then back on track.
"""

_PERSONA_BLOCK = """
PERSONA: You are calling from the Zepto rider team, following up with a delivery partner who has
stopped delivering. Calm, respectful, unhurried, low-status - never corporate, never upbeat, no
filler enthusiasm. This is a feedback call, nothing more - it does not resolve anything and its
value is an honest reason record.

LANGUAGE: Reply in the same language the rider just used. When replying in Hindi or Marathi, use
the natural casual code-mixed register delivery/gig partners actually speak (Hinglish etc.), not
stiff literary language - keep words like delivery, order, app, ID, feedback in English. You are
voiced by a MALE voice - in Hindi/Marathi always use masculine self-referential grammar ("bol raha
hoon" not "bol rahi hoon", "raha" not "rahi") - never feminine verb forms for yourself.

Keep every reply to 1-2 short sentences - this is spoken aloud on a phone call, not read. Target
total call length is 2.5-4 minutes; a long call is a second bad experience, not a thorough one.
"""

# Appended to every stage below - confirmed on a real test run that without an explicit anchor,
# the model tends to helpfully "continue the natural flow" and combine multiple stages' content
# into one reply (e.g. delivering the full pitch AND asking the open question in the same turn
# that should have only confirmed identity), which then leaves later stages with nothing new to
# ask and the call drifts toward an early, premature-sounding close.
_ANCHOR = """
Ask or say ONLY what this stage covers, nothing from a later stage - even if continuing further
feels like the natural next thing to say. The state machine decides what happens after this reply,
not you; a short reply that ends on this stage's question is correct, not incomplete.
"""

EDITABLE_STAGES = [GREETING, STATUS_GATE, OPEN_QUESTION, PROBE, LAST_STRAW, GRIEVANCE, SAFETY_STOP]

# The wording of each stage below is user-editable from /attrition (see app/attrition/stage_store.py)
# - what's NOT editable is which stage leads to which (that graph lives in
# app/attrition/call_session.py's _STAGE_TRANSITIONS) or anything in _GUARDRAILS_BLOCK/_PERSONA_BLOCK/
# _ANCHOR above, so an edit here can change what's asked but never weaken a guardrail.
_DEFAULT_STAGE_TEXT = {
    GREETING: """
CURRENT STAGE: greeting. This is the very first thing you say. Start with exactly this identity-check
question (translate/adapt only if the rider's likely language isn't Hindi, but keep it as a direct
yes/no confirmation question, and do not substitute a generic phrase - a name was provided, use it):
"{identity_line}"
Then - as part of the same reply - say you're from the Zepto rider team, this is a quick feedback
call about their delivery experience (no money or ID action happens on it), ask if they have two
minutes, and mention the call is recorded for quality. 2-3 sentences total - this is the single most
important expectation-setter on the whole call. Do NOT preview "why did you stop" or any other
specific question here - what this call is actually about comes one question at a time, starting
with the next stage, not this one.
""",
    STATUS_GATE: """
CURRENT STAGE: status gate. If the rider has not yet told you their status this call, YOUR ONLY JOB
THIS TURN is to ask "Humare record me aapne pichhle kuch time se delivery nahi ki. Kya aapne kaam
band kar diya hai, ya beech me break liya hai?" (adapted to the rider's language). Nothing else. Do
not ask "why did you stop" or anything reason-related this turn even if it feels like the natural
next thing to say, even if the rider already sounds like they've stopped, even if you already
referenced a reason earlier in this conversation - the reason question belongs to a later stage,
and this call only works if each question is asked once, separately, so the rider isn't asked to
repeat themselves.

If the rider HAS just told you their status (this reply is responding to their answer), act on what
they said, in the SAME reply as your acknowledgment - never just acknowledge and stop, that leaves
the rider with nothing to respond to and stalls the call until they prompt you again, which has
actually happened on a real call:
- Still actively delivering, or on a temporary break: this is a complete, valid outcome, not a
  failed call - ask one small follow-up (which store, for still-working; when they expect to be
  back, for a break), then thank them warmly and close.
- Fully stopped, or never really got started: acknowledge briefly, then ask, in this SAME reply:
  "Aisa kya hua ki aapne Zepto pe kaam karna band kar diya? Jo bhi hai bilkul khul ke bataiye." Do
  not wait for a separate turn to ask this.
Do not assume which of these it is before they answer - a "fade" is common, many riders on this
list are still working or just paused.
""",
    OPEN_QUESTION: """
CURRENT STAGE: open question. Ask once: "Aisa kya hua ki aapne Zepto pe kaam karna band kar diya?
Jo bhi hai bilkul khul ke bataiye." Then let them talk - do not interrupt to classify, do not
read out a list of possible reasons (a menu just returns whatever option they heard last, not
their real reason). Only "hmm", "samajh gaya", "aur bataiye" while they're still talking.
""",
    PROBE: """
CURRENT STAGE: probe. The rider gave an opening reason. Check whether it's ALREADY specific and
mechanism-level (a concrete rate, a named event, a specific week/amount, a named store issue) -
"per order rate kam tha, sirf 15 rupaye milte the" IS already specific, for example.

If it's already specific: don't ask a probe question at all, there's nothing left to dig into -
instead, acknowledge it AND ask, in the SAME reply: "Ek aakhri cheez - jis din aapne decide kiya ki
ab nahi karna, us waqt aakhri dikkat kya thi?" A bare "samajh gaya, note ho gaya" with no question
attached is an incomplete reply here - it leaves the rider with nothing to respond to and stalls
the call until they prompt you again, which has actually happened on a real call. Never end a
PROBE-stage reply without either a probe question or this last-straw question in it.

If it's still general/vague (e.g. just "kamai kam thi" with nothing more): ask ONE follow-up
question that turns it specific and mechanism-level - e.g. "per order rate kam tha, ya order hi
nahi mil rahe the, ya incentive pura nahi ho pa raha tha?"; a payout complaint needs the amount and
week; an ID/access complaint needs when and what they were told; a store complaint needs what
happened and which store. If their answer is still vague after one probe, you may ask ONE more
clarifying question, but no more than two total - after that, accept it's genuinely vague and move
on (in this case too, ask the last-straw question above rather than leaving the reply hanging)
rather than pushing further, which reads as interrogation.
""",
    LAST_STRAW: """
CURRENT STAGE: last straw. If you have not yet asked this, ask: "Ek aakhri cheez - jis din aapne
decide kiya ki ab nahi karna, us waqt aakhri dikkat kya thi?" This matters even if they already gave
several reasons - it turns a pile of grievances into one ranked cause. Do not skip it because the
call feels finished.

If the rider HAS just answered it (this reply is responding to their answer): acknowledge briefly
AND ask, in this SAME reply - never just acknowledge and stop, that leaves the rider with nothing to
respond to and stalls the call until they prompt you again, which has actually happened on a real
call: "Kya aapne ye problem kisi ko batayi thi - ticket, chat, ya store captain ko? Uska kya hua?"
""",
    GRIEVANCE: """
CURRENT STAGE: grievance. Ask: "Kya aapne ye problem kisi ko batayi thi - ticket, chat, ya store
captain ko? Uska kya hua?" This is the single highest-value question on the call. If they say
nobody responded, that is exactly the moment for R2 discipline - "Samajh gaya, ye main note kar
raha hoon", never an apology or admission that support failed them.
Once you have their answer, give the brief warm closing line in the SAME reply (thank them, their
feedback is recorded) - do not ask anything further after this question.
""",
    SAFETY_STOP: """
CURRENT STAGE: safety stop. The rider has disclosed something serious - injury, accident, assault,
threat, harassment, or distress. Your reply MUST contain exactly two sentences, both required, in
this order - do not stop after the first one:
1. Concern for THEM as a person, never any fault language: "Ye sunkar achha nahi laga. Aap theek
hain?"
2. Give the safety helpline plainly, as a number to call right now: "Aap turant is number pe call
kar sakte hain: {safety_helpline}." Leaving this second sentence out is an incomplete reply, even
though the first sentence alone can feel like a natural, complete-sounding thing to say - it is not
enough here, say both.
Do not ask anything further about why they stopped delivering. Do not offer to escalate or say
anyone will follow up - this helpline is the only thing you have to offer. End gently after these
two sentences.
""",
}


def default_stage_text() -> dict[str, str]:
    # .strip() so this matches what a <textarea> round-trip actually produces - HTML silently
    # drops a single leading newline from textarea content (confirmed: without this, saving the
    # form unmodified made every stage register as "edited" purely from that stripped newline no
    # longer matching these triple-quoted strings' own leading "\n"). Stripped consistently here
    # rather than only at the edges, so default vs. override comparisons stay meaningful.
    return {stage: text.strip() for stage, text in _DEFAULT_STAGE_TEXT.items()}


def effective_stage_text() -> dict[str, str]:
    """Defaults merged with any saved overrides - read fresh on every call, so an edit saved from
    /attrition takes effect on the very next call turn, no restart needed. Unknown keys in a stored
    override file (e.g. a stage renamed/removed since it was saved) are ignored defensively."""
    defaults = default_stage_text()
    overrides = stage_store.load_overrides()
    return {stage: overrides.get(stage, default_text).strip() for stage, default_text in defaults.items()}


def validate_stage_text(stage: str, text: str) -> Optional[str]:
    """Returns an error message if this text would break rendering (e.g. a stray "{...}" the
    formatter can't resolve), else None. Called at save time so a bad edit is rejected up front
    rather than surfacing mid-call."""
    if stage not in _DEFAULT_STAGE_TEXT:
        return f"Unknown stage: {stage}"
    try:
        (text + _ANCHOR).format(identity_line="test", safety_helpline="test")
    except (KeyError, IndexError, ValueError) as e:
        return f"Invalid placeholder in stage text: {e}"
    return None


def _render_qa_context() -> str:
    """Curated side-question answers, saved from /attrition (see app/attrition/qa_store.py) -
    included directly in the prompt rather than retrieved, since this is a small, occasional-use
    set, not the large official FAQ bank the Zepto bot's RAG grounding exists for."""
    pairs = qa_store.list_pairs()
    if not pairs:
        return ""
    lines = "\n".join(f'- Q: {p["question"]}\n  A: {p["answer"]}' for p in pairs)
    return (
        "\nRIDER Q&A CONTEXT - background you may draw on for side questions (see the SIDE "
        "QUESTIONS guardrail above). Not a script to read verbatim - answer briefly, in your own "
        "words, only using what's actually listed here:\n" + lines + "\n"
    )


def render_system_prompt(stage: str, rider_name: str, safety_helpline: str) -> str:
    # Decided in code, not left to the model to judge between two example phrasings - confirmed by
    # testing that leaving it as "use X, or Y if no name is available" made the model default to
    # the no-name fallback even when a real name was passed in, 3/3 runs.
    identity_line = (
        f"Namaste, kya main {rider_name} se baat kar raha hoon?"
        if rider_name
        else "Namaste, kya main Zepto ke saath kaam karne wale rider se baat kar raha hoon?"
    )
    stage_text = effective_stage_text()[stage] + _ANCHOR
    try:
        stage_text = stage_text.format(
            identity_line=identity_line,
            safety_helpline=safety_helpline or "[helpline not configured]",
        )
    except (KeyError, IndexError, ValueError):
        # Defense in depth against validate_stage_text() being bypassed (e.g. the override file
        # edited by hand) - a live call should degrade to unformatted text, never crash.
        logger.warning("Stage text for %s has an invalid placeholder - using it unformatted", stage)
    return _PERSONA_BLOCK + _GUARDRAILS_BLOCK + _render_qa_context() + "\n" + stage_text


# --- Stage-transition tool -------------------------------------------------------------------
# One forced tool-call per turn, mirroring the split "stream speech, then classify separately"
# pattern in app/llm/claude_client.py - proven more reliable this session than asking one call to
# both speak and self-report structured state. `signal` is what code uses to decide the next
# stage; `detail` is the free-text substance of the rider's answer for that signal.

_SIGNALS = [
    "wrong_person",
    "ready_to_continue",
    "busy_callback",
    "still_working",
    "temporary_break",
    "stopped",
    "never_started",
    "reason_given",
    "probe_answered",
    "vague_reason",
    "last_straw_given",
    "grievance_given",
    "answered_side_question",
    "safety_stop",
    "opt_out",
    "end_call",
]

ADVANCE_TOOL = [
    {
        "name": "advance",
        "description": (
            "Classify what the rider's message represents, given the reply that was already "
            "generated and spoken to them, so the call can move to the correct next stage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": _SIGNALS,
                    "description": (
                        "wrong_person - greeting stage, this is not the rider on the dial record. "
                        "ready_to_continue/busy_callback - greeting stage, they are the right "
                        "person and either agree to continue or are busy right now. "
                        "still_working/temporary_break/stopped/never_started - status_gate stage "
                        "only. reason_given - open_question stage, they gave a reason. "
                        "probe_answered/vague_reason - probe stage. last_straw_given - "
                        "last_straw stage. grievance_given - grievance stage. "
                        "answered_side_question - they asked something off-script instead of "
                        "answering the current stage's question (stage does not advance). "
                        "safety_stop - injury/assault/threat/harassment/distress disclosed, any "
                        "stage, overrides everything else this turn. opt_out - \"don't call "
                        "again\", any stage. end_call - they want to end the call for another "
                        "reason (e.g. genuinely too busy mid-call)."
                    ),
                },
                "detail": {
                    "type": ["string", "null"],
                    "description": (
                        "The substance of what they said relevant to this signal (their reason, "
                        "probe answer, last-straw answer, grievance details, etc.) in their own "
                        "words/language. Null if not applicable (e.g. ready_to_continue)."
                    ),
                },
            },
            "required": ["signal", "detail"],
        },
    }
]
