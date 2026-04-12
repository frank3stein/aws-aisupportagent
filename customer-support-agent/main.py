"""
Customer Support AI Agent
=========================
Run locally:
  uv run --env-file .env main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, asyncio, boto3, argparse, json, uuid, math, logging
from datetime import datetime, timezone
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import tool
from strands.hooks import HookProvider, AfterInvocationEvent, MessageAddedEvent
from typing import Dict
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── Configuration ─────────────────────────────────────────────────────────────
GATEWAY_URL = os.environ.get("GATEWAY_URL")
KB_ID       = os.environ.get("KB_ID")
REGION      = os.environ.get("REGION", "us-east-1")
MEMORY_ID   = os.environ.get("MEMORY_ID")

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── App — must be module-level, AgentCore discovers it here ───────────────────
app = BedrockAgentCoreApp()

# ── Lazy singletons ───────────────────────────────────────────────────────────
# Kept None until the first real request to keep startup under 30s.
_model           = None
_memory_client   = None
_bedrock_runtime = None


def _get_clients():
    """Initialise heavy clients once, on the first invocation."""
    global _model, _memory_client, _bedrock_runtime
    if _model is None:
        from strands.models import BedrockModel
        from bedrock_agentcore.memory import MemoryClient
        logger.info("Initialising clients...")
        _model           = BedrockModel(model_id="global.amazon.nova-2-lite-v1:0")
        _memory_client   = MemoryClient(region_name=REGION)
        _bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
        logger.info("Clients ready.")
    return _model, _memory_client, _bedrock_runtime


# ── Namespace Helper ──────────────────────────────────────────────────────────
def get_namespaces(mem_client, memory_id: str) -> Dict:
    """Return a dict mapping strategy type to namespace template string.

    Example:
      { "SEMANTIC": "cs_agent/{actorId}/facts",
        "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }
    """
    strategies = mem_client.get_memory_strategies(memory_id)
    return {
        strategy["type"]: strategy["namespaces"][0]
        for strategy in strategies
    }


# ── Memory Hook ───────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Adds long-term memory retrieval and persistence to the agent."""

    def __init__(self, actor_id: str, session_id: str,
                 memory_client, memory_id: str):
        self.actor_id      = actor_id
        self.session_id    = session_id
        self.memory_client = memory_client
        self.memory_id     = memory_id
        self.namespaces    = get_namespaces(memory_client, memory_id)

    def register_hooks(self, registry) -> None:
        # Type annotations on the methods tell add_hook which event to bind to
        logger.warning(f"Registry type: {type(registry)}, attrs: {[a for a in dir(registry) if not a.startswith('_')]}")
        registry.add_hook(self.retrieve_customer_context)
        registry.add_hook(self.save_support_interaction)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        logger.warning(f"retrieve_customer_context FIRED for {self.actor_id}")
        messages = event.agent.messages
        if not messages:
            return
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        content = last_message.get("content", "")

        # Extract text from both string and list-of-dicts formats
        if isinstance(content, str):
            user_query = content.strip()
        elif isinstance(content, list):
            user_query = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ).strip()
        else:
            return

        if not user_query:
            return

        logger.warning(f"Querying memory with: {user_query[:50]}")
        memory_lines = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.replace("{actorId}", self.actor_id)
            logger.warning(f"Querying namespace: {namespace}")
            try:
                results = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
                logger.warning(f"Results: {results}")
                for mem in (results or []):
                    text = mem.get("content", {}).get("text", "").strip()
                    if text:
                        memory_lines.append(f"[{strategy_type}] {text}")
            except Exception as e:
                logger.warning(f"Memory retrieval failed for {strategy_type}: {e}")

        if memory_lines:
            context_block = "Customer Context:\n" + "\n".join(memory_lines)
            for block in last_message["content"]:
                if isinstance(block, dict) and "text" in block:
                    block["text"] = f"{context_block}\n\n{block['text']}"
                    logger.warning(f"Inserted context into block: {block}")
                    break
            else:
                last_message["content"] = f"{context_block}\n\n{user_query}"
            logger.warning(f"Message after prepend: {last_message}")  # add this

    def save_support_interaction(self, event: AfterInvocationEvent):
        logger.warning(f"save_support_interaction FIRED for {self.actor_id}")
        messages       = event.agent.messages
        customer_query = None
        agent_response = None

        for msg in reversed(messages):
            role    = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("text")
                ).strip()
            else:
                text = ""

            if text:
                if role == "assistant" and agent_response is None:
                    agent_response = text
                elif role == "user" and customer_query is None:
                    customer_query = text

            if customer_query and agent_response:
                break

        if customer_query and agent_response:
            try:
                from datetime import datetime, timezone
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (customer_query, "USER"),
                        (agent_response, "ASSISTANT"),
                    ],
                )
                logger.warning(f"Memory saved for {self.actor_id}")
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")
# ── Knowledge Base Tool ───────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured. Set KB_ID as an environment variable."
    _, _, bedrock_runtime = _get_clients()
    try:
        resp = bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."
        chunks = [
            r["content"]["text"]
            for r in results
            if r.get("content", {}).get("text")
        ]
        return "\n---\n".join(chunks)
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return f"Knowledge base search failed: {str(e)}"


# ── Loyalty Discount Tool ─────────────────────────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price as a JSON string
    """
    from bedrock_agentcore.tools.code_interpreter_client import code_session  # lazy

    code = f"""
import math, json

loyalty_points   = {loyalty_points}
tier             = "{tier}"
order_total      = {order_total}
product_category = "{product_category}"

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

max_redeemable     = math.floor((order_total * 0.5) / 500) * 500
points_redeemed    = min(math.floor(loyalty_points / 500) * 500, max_redeemable)
points_value       = points_redeemed * 0.01

subtotal_after_pts = order_total - points_value
tier_discount_rate = tier_rates.get(tier, 0.0)
tier_discount      = subtotal_after_pts * tier_discount_rate
final_total        = subtotal_after_pts - tier_discount

earn_rate          = earn_rates.get(product_category, 1)
points_earned      = int(final_total * earn_rate)
remaining_points   = loyalty_points - points_redeemed + points_earned
total_savings      = points_value + tier_discount

result = {{
    "points_redeemed":    points_redeemed,
    "points_value_usd":   round(points_value, 2),
    "tier":               tier,
    "tier_discount_rate": tier_discount_rate,
    "tier_discount_usd":  round(tier_discount, 2),
    "final_total":        round(final_total, 2),
    "total_savings":      round(total_savings, 2),
    "points_earned":      points_earned,
    "remaining_points":   remaining_points,
}}
print(json.dumps(result))
"""
    try:
        with code_session(REGION) as session:
            response = session.invoke("executeCode", {
                "language": "python",
                "code": code,
                "clearContext": True,
            })
            events = response.get("result", [])
            if events:
                return json.dumps(events[0])
            return json.dumps(response)
    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback: {e}")

    # Fallback: tier-only discount
    tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
    rate       = tier_rates.get(tier, 0.0)
    discount   = order_total * rate
    final      = order_total - discount
    return json.dumps({
        "tier":               tier,
        "tier_discount_rate": rate,
        "tier_discount_usd":  round(discount, 2),
        "final_total":        round(final, 2),
        "note": "Fallback calculation — Code Interpreter unavailable",
    })


# ── Agent Entrypoint ──────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    # Copy Playwright node binary to /tmp and make it executable
    # /var/task is read-only in AgentCore container, /tmp is writable
    import stat, shutil
    src = "/var/task/playwright/driver/node"
    dst = "/tmp/playwright_node"
    if os.path.exists(src) and not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)
            os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            os.environ["PLAYWRIGHT_NODEJS_PATH"] = dst
            logger.warning(f"Copied playwright node to {dst}")
        except Exception as e:
            logger.warning(f"Could not copy playwright node: {e}")
    elif os.path.exists(dst):
        os.environ["PLAYWRIGHT_NODEJS_PATH"] = dst
        logger.warning(f"Using existing playwright node at {dst}")
    from strands import Agent
    from strands.tools.mcp.mcp_client import MCPClient
    from mcp.client.streamable_http import streamable_http_client
    from strands_tools.browser import AgentCoreBrowser  

    logger.warning(f"Raw payload: {payload}")

    if "customer_id" not in payload and "prompt" in payload:
        try:
            inner = json.loads(payload["prompt"])
            payload = inner
        except Exception:
            pass

    user_input = payload.get("prompt", "")
    actor_id   = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id", str(uuid.uuid4()))
    logger.warning(f"actor_id={actor_id} session_id={session_id}")

    try:
        model, memory_client, _ = _get_clients()

        memory_hook        = MemoryHook(actor_id, session_id, memory_client, MEMORY_ID)
        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        with MCPClient(lambda: streamable_http_client(GATEWAY_URL)) as mcp:
            gateway_tools = mcp.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=(
                    "You are a helpful customer support assistant for an e-commerce platform. "
                    "You can track orders, process refunds, answer product questions, "
                    "calculate loyalty discounts, and browse the web for current information. "
                    "Always be polite, concise, and accurate.\n\n"
                    "IMPORTANT: When the user message begins with 'Customer Context:', "
                    "that section contains verified facts and preferences about this customer "
                    "retrieved from long-term memory. You MUST use this information to "
                    "personalise your response. For example, if the context says the user's "
                    "name is Jane, address them as Jane. If it says they prefer concise "
                    "responses, keep your responses short. Never tell the customer you cannot "
                    "remember them if their context has been provided."
                ),
            )

            from strands.hooks import MessageAddedEvent, AfterInvocationEvent
            agent.hooks.add_callback(MessageAddedEvent,    memory_hook.retrieve_customer_context)
            agent.hooks.add_callback(AfterInvocationEvent, memory_hook.save_support_interaction)
            logger.warning("Memory hooks registered via add_callback")

            response = await agent.invoke_async(user_input)
            logger.warning(f"response.message: {response.message}")

        try:
            msg = response.message
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            return block["text"]
            return str(response)
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return str(response)

    except Exception as e:  # ← this was missing
        logger.error(f"invoke() error: {e}")
        return f"I encountered an error processing your request: {str(e)}"

# ── Local CLI entry ───────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    import io, contextlib
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    # Suppress streaming stdout from AgentResult during invoke
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # For local CLI testing, comment app.run() above and uncomment below:
    # main()