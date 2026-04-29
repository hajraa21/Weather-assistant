"""
weather_app.py
==============
Streamlit frontend for the LangGraph + Groq weather assistant.

Run with:
    streamlit run weather_app.py

Requirements (add to your requirements.txt / pip install):
    streamlit
    langchain-groq
    langgraph
    langchain-mcp-adapters
    langchain-core
    python-dotenv
"""

# ── Standard library ──────────────────────────────────────────────────────────
import asyncio
import time
from datetime import datetime

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# ── Load environment variables (.env file) ────────────────────────────────────
load_dotenv()


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the FIRST Streamlit call)
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Weather AI Assistant",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  – dark, weather-themed look
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0d1f2d 100%);
    color: #e8f4fd;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f2d 0%, #1a2a3a 100%);
    border-right: 1px solid rgba(100,180,255,0.15);
}
[data-testid="stSidebar"] * { color: #c8e6f7 !important; }

/* ── Chat message bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1e4d7a, #2563a8);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    border: 1px solid rgba(100,180,255,0.25);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    font-size: 0.95rem;
    line-height: 1.55;
    color: #e8f4fd;
}

.ai-bubble {
    background: linear-gradient(135deg, #142233, #1c3347);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    border: 1px solid rgba(100,180,255,0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    font-size: 0.95rem;
    line-height: 1.6;
    color: #d0ebfa;
}

/* ── Avatar labels ── */
.avatar-user { text-align: right; font-size: 0.72rem; color: #90caf9; margin-bottom: 3px; }
.avatar-ai   { text-align: left;  font-size: 0.72rem; color: #64b5f6; margin-bottom: 3px; }

/* ── Timestamp ── */
.msg-time {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    opacity: 0.45;
    margin-top: 5px;
}
.msg-time-right { text-align: right; }

/* ── Input area ── */
.stTextInput > div > div > input {
    background: rgba(20, 40, 60, 0.8) !important;
    border: 1px solid rgba(100,180,255,0.3) !important;
    border-radius: 12px !important;
    color: #e8f4fd !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #64b5f6 !important;
    box-shadow: 0 0 0 2px rgba(100,180,255,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #5a8aaa !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(30,136,229,0.4) !important;
}

/* ── Dividers ── */
hr { border-color: rgba(100,180,255,0.15) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(20,40,60,0.6);
    border: 1px solid rgba(100,180,255,0.15);
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stMetricValue"] { color: #64b5f6 !important; }
[data-testid="stMetricLabel"] { color: #90caf9 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(20,40,60,0.4);
    border: 1px solid rgba(100,180,255,0.12);
    border-radius: 10px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64b5f6;
    opacity: 0.7;
    margin-bottom: 8px;
}

/* ── Quick-chip buttons ── */
.stButton.chip > button {
    background: rgba(30, 80, 130, 0.5) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
#  st.session_state persists across Streamlit reruns within the same browser tab.
# ════════════════════════════════════════════════════════════════════════════════
def init_session():
    """Initialise all session-state keys if they don't exist yet."""

    # Full conversation: list of {"role": "user"|"assistant", "content": str, "time": str}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Whether the graph + tools have been compiled this session
    if "graph_ready" not in st.session_state:
        st.session_state.graph_ready = False
        st.session_state.graph      = None

    # Simple counter so we can show "N queries answered" in the sidebar
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    # Stores the last raw LangGraph output for the debug expander
    if "last_raw_output" not in st.session_state:
        st.session_state.last_raw_output = None

    # User-configurable settings
    if "show_timestamps" not in st.session_state:
        st.session_state.show_timestamps = True

    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    if "groq_model" not in st.session_state:
        st.session_state.groq_model = "llama-3.3-70b-versatile"

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a helpful assistant with access to weather tools. "
            "Always use the tools to provide accurate real-time weather data "
            "in both Celsius and Fahrenheit. Do not make assumptions; provide "
            "only genuine info to the user. Also provide useful lifestyle tips "
            "based on the weather such as whether it is a good day to go out, "
            "what kind of clothes to wear, what food will be good in this weather, etc."
        )


# ════════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH SETUP  – compiled once per session and cached in session_state
# ════════════════════════════════════════════════════════════════════════════════

async def build_graph(model_name: str, system_prompt: str):
    """
    Connect to the MCP weather server, bind tools to the LLM,
    and compile the LangGraph state machine.

    This is called once and the compiled graph is stored in
    st.session_state.graph so it can be reused across queries.
    """

    # ── 1. Connect to the MCP weather server ──────────────────────────────────
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "https://weather-d26bfd98e65f.fastmcp.app/mcp",
                "headers": {
                    # Replace with your actual Bearer token if needed
                    "Authorization": "Bearer fmcp_hCVUlCHnEp4fokPkVlLO9CcqTKzdi7U9K3th4sgBATo"
                },
            }
        }
    )
    tools = await client.get_tools()

    # ── 2. Initialise the LLM and bind weather tools ──────────────────────────
    llm = ChatGroq(model=model_name)
    llm_with_tools = llm.bind_tools(tools)

    # ── 3. Define the graph state schema ─────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # ── 4. Chat node: injects system prompt and calls the LLM ─────────────────
    async def chat_node(state: State):
        messages = state["messages"]
        sys_msg  = SystemMessage(content=system_prompt)

        # Prepend system message if it isn't already there
        if not isinstance(messages[0], SystemMessage):
            messages = [sys_msg] + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # ── 5. Wire up the graph ──────────────────────────────────────────────────
    g = StateGraph(State)
    g.add_node("chat_node", chat_node)
    g.add_node("tools", ToolNode(tools=tools))

    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)  # → tools OR END
    g.add_edge("tools", "chat_node")                       # tool result loops back

    return g.compile()


def get_or_build_graph():
    """
    Return the cached compiled graph, building it if necessary.
    Wraps the async builder with asyncio.run() so Streamlit (sync) can call it.
    """
    if not st.session_state.graph_ready:
        with st.spinner("🔧 Connecting to weather service and compiling AI graph…"):
            try:
                graph = asyncio.run(
                    build_graph(
                        st.session_state.groq_model,
                        st.session_state.system_prompt,
                    )
                )
                st.session_state.graph      = graph
                st.session_state.graph_ready = True
            except Exception as e:
                st.error(f"❌ Failed to initialise graph: {e}")
                return None

    return st.session_state.graph


# ════════════════════════════════════════════════════════════════════════════════
#  QUERY RUNNER
# ════════════════════════════════════════════════════════════════════════════════

async def run_query(graph, conversation_history: list) -> str:
    """
    Convert the Streamlit chat history into LangChain messages,
    pass them to the compiled graph, and return the final AI reply.

    Parameters
    ----------
    graph               : compiled LangGraph StateGraph
    conversation_history: list of {"role": ..., "content": ...}

    Returns
    -------
    str  – the assistant's reply text
    """

    # Build the LangChain message list from the stored conversation
    lc_messages: list[BaseMessage] = []
    for msg in conversation_history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    output = await graph.ainvoke({"messages": lc_messages})

    # Store raw output for the debug panel
    st.session_state.last_raw_output = output

    # Return the content of the last message in the graph output
    return output["messages"][-1].content


def send_message(user_input: str):
    """
    Called when the user submits a query.
    Appends user + assistant messages to session history and invokes the graph.
    """
    if not user_input.strip():
        return

    graph = get_or_build_graph()
    if graph is None:
        return

    # ── Record the user message ───────────────────────────────────────────────
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "time": now}
    )

    # ── Run the graph (sync wrapper around async) ─────────────────────────────
    with st.spinner("🌐 Fetching weather data…"):
        try:
            start = time.time()
            reply = asyncio.run(run_query(graph, st.session_state.messages))
            elapsed = round(time.time() - start, 2)
        except Exception as e:
            reply   = f"⚠️ Something went wrong: {e}"
            elapsed = 0

    # ── Record the assistant message ──────────────────────────────────────────
    st.session_state.messages.append(
        {
            "role":    "assistant",
            "content": reply,
            "time":    datetime.now().strftime("%H:%M"),
            "elapsed": elapsed,
        }
    )

    st.session_state.query_count += 1


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown("## 🌤️ Weather AI")
        st.markdown(
            "<span style='font-size:0.8rem;opacity:0.6'>"
            "Powered by Groq · LangGraph · MCP</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Stats ─────────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Session Stats</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Queries", st.session_state.query_count)
        col2.metric("Messages", len(st.session_state.messages))
        st.divider()

        # ── Quick queries ─────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Quick Queries</p>', unsafe_allow_html=True)

        quick_queries = [
            "🌆 Weather in Mumbai",
            "❄️  Weather in London",
            "☀️  Weather in Dubai",
            "🌧️ Weather in Singapore",
            "🌨️ Weather in New York",
            "🌞 Weather in Sydney",
        ]

        for q in quick_queries:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                # Strip the emoji prefix to form the actual query
                city_part = q.split("  ")[-1].strip() if "  " in q else q[2:].strip()
                send_message(city_part)
                st.rerun()

        st.divider()

        # ── Settings ──────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)

        # Groq model selector
        model_choice = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
            key="groq_model_selector",
            help="Choose the underlying language model.",
        )

        # If model changed, force graph rebuild on next query
        if model_choice != st.session_state.groq_model:
            st.session_state.groq_model  = model_choice
            st.session_state.graph_ready = False
            st.info("Graph will rebuild on your next query.")

        st.session_state.show_timestamps = st.toggle(
            "Show timestamps", value=st.session_state.show_timestamps
        )
        st.session_state.show_debug = st.toggle(
            "Show debug panel", value=st.session_state.show_debug
        )

        st.divider()

        # ── Custom system prompt ───────────────────────────────────────────────
        with st.expander("⚙️ System Prompt"):
            new_prompt = st.text_area(
                "Edit the assistant's behaviour",
                value=st.session_state.system_prompt,
                height=160,
                label_visibility="collapsed",
            )
            if st.button("Apply & Rebuild Graph"):
                st.session_state.system_prompt  = new_prompt
                st.session_state.graph_ready    = False
                st.success("Prompt updated. Graph will rebuild on next query.")

        st.divider()

        # ── Clear conversation ────────────────────────────────────────────────
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages        = []
            st.session_state.last_raw_output = None
            st.session_state.query_count     = 0
            st.rerun()

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            "<br><span style='font-size:0.7rem;opacity:0.4'>"
            "LangGraph · LangChain MCP · ChatGroq</span>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY RENDERER
# ════════════════════════════════════════════════════════════════════════════════

def render_chat():
    """Render every message in the conversation history as styled HTML bubbles."""

    if not st.session_state.messages:
        # ── Empty-state placeholder ───────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px; opacity: 0.5;">
                <div style="font-size: 3rem;">🌤️</div>
                <div style="font-size: 1.1rem; margin-top: 12px;">
                    Ask me about the weather anywhere in the world.
                </div>
                <div style="font-size: 0.85rem; margin-top: 8px; opacity: 0.7;">
                    I'll fetch real-time data and give you personalised tips.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        ts      = msg.get("time", "")
        elapsed = msg.get("elapsed", None)

        if role == "user":
            # Right-aligned user bubble
            time_html = (
                f'<div class="msg-time msg-time-right">{ts}</div>'
                if st.session_state.show_timestamps else ""
            )
            st.markdown(
                f"""
                <div class="avatar-user">You 👤</div>
                <div class="user-bubble">{content}</div>
                {time_html}
                """,
                unsafe_allow_html=True,
            )

        else:
            # Left-aligned AI bubble
            elapsed_str = f" · {elapsed}s" if elapsed else ""
            time_html = (
                f'<div class="msg-time">{ts}{elapsed_str}</div>'
                if st.session_state.show_timestamps else ""
            )
            st.markdown(
                f"""
                <div class="avatar-ai">🌤️ Weather AI</div>
                <div class="ai-bubble">{content}</div>
                {time_html}
                """,
                unsafe_allow_html=True,
            )

    # ── Debug: raw LangGraph output ───────────────────────────────────────────
    if st.session_state.show_debug and st.session_state.last_raw_output:
        with st.expander("🔍 Raw LangGraph Output (debug)"):
            st.json(
                {
                    "message_count": len(
                        st.session_state.last_raw_output.get("messages", [])
                    ),
                    "last_message_type": str(
                        type(
                            st.session_state.last_raw_output["messages"][-1]
                        ).__name__
                    ),
                    "last_content": st.session_state.last_raw_output["messages"][
                        -1
                    ].content[:500],
                }
            )


# ════════════════════════════════════════════════════════════════════════════════
#  INPUT AREA
# ════════════════════════════════════════════════════════════════════════════════

def render_input():
    """Render the text input + send button at the bottom of the page."""

    st.divider()
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_input = st.text_input(
            label="Ask about the weather",
            placeholder="e.g. What's the weather like in Tokyo right now?",
            label_visibility="collapsed",
            key="user_input_field",
        )

    with col_btn:
        send_clicked = st.button("Send ➤", use_container_width=True)

    # Trigger on button click OR pressing Enter (non-empty field)
    if send_clicked and user_input:
        send_message(user_input)
        st.rerun()  # refresh so the new messages appear immediately


# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════

def render_header():
    col_title, col_status = st.columns([4, 1])

    with col_title:
        st.markdown(
            "<h1 style='margin:0; font-size:1.8rem;'>🌤️ Weather AI Assistant</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='margin:0; opacity:0.55; font-size:0.85rem;'>"
            "Real-time weather data · Personalised lifestyle tips · Powered by Groq + LangGraph"
            "</p>",
            unsafe_allow_html=True,
        )

    with col_status:
        if st.session_state.graph_ready:
            st.markdown(
                "<div style='text-align:right; color:#66bb6a; font-size:0.8rem; margin-top:14px;'>"
                "● Connected</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:right; color:#ef9a9a; font-size:0.8rem; margin-top:14px;'>"
                "○ Not connected</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    init_session()      # Make sure all session keys exist
    render_sidebar()    # Left panel: stats, quick queries, settings
    render_header()     # Top: title + connection status
    render_chat()       # Middle: conversation history
    render_input()      # Bottom: text box + send button

    # Pre-warm the graph on first load so the first query is fast
    if not st.session_state.graph_ready:
        get_or_build_graph()


if __name__ == "__main__":
    main()