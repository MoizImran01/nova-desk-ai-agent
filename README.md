# Nova Wellness & Aesthetics - AI Concierge System

Welcome to the backend engine of Nova Desk, a luxury medical spa AI assistant. Powered by FastAPI and LangGraph, this backend orchestrates intent classification, dynamic scheduling, multi-turn booking/rescheduling, and conversational FAQ retrievals with enterprise-grade guardrails.
## 1. Backend Agent Engine (`back-end/`)

Powered by **FastAPI** and **LangGraph**, the backend hosts the conversational state machine, memory persistence, fuzzy matching systems, and DB-connected scheduling tools.

* **Vercel Deployed URL:** [https://nova-desk-ai-agent.vercel.app](https://nova-desk-ai-agent.vercel.app)

## 🚀 Key Features

* **Multi-Agent Orchestration**: Structured state machine using **LangGraph** to manage complex conversational paths, conditional routing, and cyclic loops.
* **Intelligent Intent Classification**: Dynamic classification of user messages into distinct flows (`faq`, `appointment`, `appointment_reschedule`, `human_escalation`, `out_of_scope`).
* **Robust Out-of-Scope Guardrails**: Zero-cost deterministic prompt filter bypassing the LLM for off-topic prompts to prevent jailbreaking/hijacking and save API tokens.
* **Context-Aware Booking**: Multi-turn dialogue collection for appointment details (Name, Email, Service, Date, Time) with validation.
* **Auto-Alternative Slot Finder**: Gracefully handle fully-booked slots by searching database schedules up to 7 days ahead and proposing nearest alternatives.
* **Fuzzy Service Matching**: Corrects user typos for service names (e.g., matching "botoxx" to "Botox") automatically using Levenshtein distance.
* **Possessive Identification**: Distinguishes Selection (*"Let's change Monday's appointment"*) from Rescheduling (*"Move the appointment to Monday"*).
* **Relative Date Math & Resolution**: Processes complex expressions like *"next day"*, *"tomorrow"*, or *"next Friday"* relative to target dates.
* **Date Ambiguity Disambiguation**: Holds the execution state to clarify ambiguous inputs (like just *"Friday"*) via conversational confirmation loops.
* **Tool Call Hallucination Fallback**: Safely catches LLM schema hallucinations, transitioning back to natural text chains without crashing.

---

## 🏗️ System Architecture & State Flow

```mermaid
graph TD
    %% Styling Classes
    classDef startEnd fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    classDef router fill:#ffebcc,stroke:#f8a100,stroke-width:1.5px,stroke-dasharray: 5 5,color:#000;
    classDef node fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef tool fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef db fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#000;

    %% Entry and Exit Points
    START([START]) --> classify_intent
    class START startEnd;

    %% Persistence & Database
    subgraph Storage ["Persistence & Data Layer"]
        Database[("PostgreSQL Database<br/>(Users, Appointments, Conversations)")]
        Checkpointer[("PostgresSaver Checkpointer<br/>(LangGraph State Checkpoints)")]
    end
    class Database,Checkpointer db;

    %% Main Routing Area
    subgraph IntentSystem ["Intent Classification & Global Routing"]
        classify_intent("classify_intent Node<br/>(Analyze latest human message)")
        route_intent{"route_intent<br/>Conditional Edge"}
        
        classify_intent --> route_intent
        
        %% Static Guardrails
        handle_out_of_scope("handle_out_of_scope Node<br/>(Fixed Rejection Message)")
        handle_retreive_faqs("handle_retreive_faqs Node<br/>(Retrieves from KB)")
        
        route_intent -.->|"out_of_scope"| handle_out_of_scope
        route_intent -.->|"faq"| handle_retreive_faqs
        
        handle_out_of_scope --> END([END])
        handle_retreive_faqs --> END
    end
    class classify_intent,handle_out_of_scope,handle_retreive_faqs node;
    class route_intent router;
    class END startEnd;

    %% Link routes from Global Router to specific flows
    route_intent -.->|"appointment"| collect_appointment_details
    route_intent -.->|"appointment_reschedule (no email/lookup)"| collect_reschedule_intent
    route_intent -.->|"appointment_reschedule (lookup done, changes incomplete)"| collect_modification_details
    route_intent -.->|"appointment_reschedule (all details collected)"| handle_apply_modification

    %% Booking Flow
    subgraph BookingFlow ["Booking Flow"]
        collect_appointment_details("collect_appointment_details Node<br/>(Extracts details)")
        route_after_collecting_details{"route_after_collecting_details<br/>Conditional Edge"}
        ask_date_confirmation("ask_date_confirmation Node<br/>(Clarify Ambiguous Date)")
        ask_for_missing_field("ask_for_missing_field Node<br/>(Request missing fields)")
        handle_appointment_booking("handle_appointment_booking Node<br/>(Handles booking check slots)")
        
        collect_appointment_details --> route_after_collecting_details
        route_after_collecting_details -.->|"ask_date_confirmation"| ask_date_confirmation
        route_after_collecting_details -.->|"ask_for_missing_field"| ask_for_missing_field
        route_after_collecting_details -.->|"handle_appointment_booking"| handle_appointment_booking
        
        ask_date_confirmation --> END
        ask_for_missing_field --> END
        
        booking_tools_condition{"tools_condition<br/>(Check for tool calls)"}
        handle_appointment_booking --> booking_tools_condition
        booking_tools_condition -.->|"No Tool Calls"| END
        booking_tools_condition -.->|"Has Tool Calls"| booking_tools
    end
    class collect_appointment_details,ask_date_confirmation,ask_for_missing_field,handle_appointment_booking node;
    class route_after_collecting_details,booking_tools_condition router;

    %% Rescheduling Flow
    subgraph RescheduleFlow ["Rescheduling Flow"]
        %% Phase 1 & 2: Email Lookup
        collect_reschedule_intent("collect_reschedule_intent Node<br/>(Extract email)")
        route_after_collecting_reschedule_intent{"route_after_collecting_reschedule_intent<br/>Conditional Edge"}
        ask_for_email("ask_for_email Node<br/>(Prompt for email)")
        handle_appointment_lookup("handle_appointment_lookup Node<br/>(Check user appointments)")
        
        collect_reschedule_intent --> route_after_collecting_reschedule_intent
        route_after_collecting_reschedule_intent -.->|"ask_for_email"| ask_for_email
        route_after_collecting_reschedule_intent -.->|"handle_appointment_lookup"| handle_appointment_lookup
        
        ask_for_email --> END
        
        lookup_tools_condition{"tools_condition<br/>(Check for tool calls)"}
        handle_appointment_lookup --> lookup_tools_condition
        lookup_tools_condition -.->|"No Tool Calls"| END
        lookup_tools_condition -.->|"Has Tool Calls"| reschedule_tools

        %% Phase 3 & 4: Modification Details & Apply
        collect_modification_details("collect_modification_details Node<br/>(Extract target changes)")
        route_after_collecting_modification{"route_after_collecting_modification<br/>Conditional Edge"}
        handle_apply_modification("handle_apply_modification Node<br/>(Executes modification turn)")
        
        collect_modification_details --> route_after_collecting_modification
        route_after_collecting_modification -.->|"ask_date_confirmation"| ask_date_confirmation
        route_after_collecting_modification -.->|"handle_apply_modification"| handle_apply_modification
        
        apply_tools_condition{"tools_condition<br/>(Check for tool calls)"}
        handle_apply_modification --> apply_tools_condition
        apply_tools_condition -.->|"No Tool Calls"| END
        apply_tools_condition -.->|"Has Tool Calls"| reschedule_tools
    end
    class collect_reschedule_intent,handle_appointment_lookup,collect_modification_details,handle_apply_modification,ask_for_email node;
    class route_after_collecting_reschedule_intent,route_after_collecting_modification,lookup_tools_condition,apply_tools_condition router;

    %% Tools Execution Layer
    subgraph ToolNodeGroup ["Tool Execution Layer"]
        booking_tools[["booking_tools ToolNode<br/>- check_available_appointment_slots<br/>- book_appointment"]]
        reschedule_tools[["reschedule_tools ToolNode<br/>- get_appointments_by_email<br/>- check_available_appointment_slots<br/>- modify_appointment"]]
        
        booking_tools -->|Returns ToolMessage| handle_appointment_booking
        
        route_after_reschedule_tools{"route_after_reschedule_tools<br/>Conditional Edge"}
        reschedule_tools --> route_after_reschedule_tools
        route_after_reschedule_tools -.->|"get_appointments_by_email"| handle_appointment_lookup
        route_after_reschedule_tools -.->|"other tools"| handle_apply_modification
    end
    class booking_tools,reschedule_tools tool;
    class route_after_reschedule_tools router;

    %% External Tool to DB Connections
    booking_tools -.->|Writes/Queries| Database
    reschedule_tools -.->|Writes/Queries| Database
    
    %% Graph Session checkpoint connection
    START -.->|Loads State| Checkpointer
    END -.->|Saves State| Checkpointer
```

---

## 🛠️ Technology Stack (Backend)

| Component | Technology | Description / Usage |
| :--- | :--- | :--- |
| **API Core** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance, asynchronous REST API router |
| **State Machine** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Handles conversation state, routing conditional edges, and loops |
| **LLM Interface** | [LangChain Core](https://github.com/langchain-ai/langchain) | Tool binding, structured outputs, and prompt templating |
| **Database ORM** | [SQLModel](https://sqlmodel.tiangolo.com/) | Modern Python SQL database wrapper combining SQLAlchemy & Pydantic |
| **Fuzzy Matching** | `difflib` (Python Standard Lib) | Levenshtein ratio matching for spa treatment validation |
| **Timezone Management** | `zoneinfo` | Standardized scheduling using local zone `Asia/Karachi` |

---

## 🌐 Hosting & Services (Backend)

* **State & Transactional DB (PostgreSQL)**: Hosted serverless on [Neon Tech](https://neon.tech/) (AWS `us-east-2`), utilizing connection poolers for high scalability.
* **Vector Database (RAG)**: Hosted on [Pinecone Serverless](https://www.pinecone.io/) using `cosine` similarity to store and query highly optimized embeddings.
* **Large Language Models (LLMs)**:
  * **Groq API**: High-speed LLaMA models for primary intent classification and text generation.
  * **Google Gemini API**: Utilizing `text-embedding-004` to generate 768-dimensional document vectors, alongside large-context models for complex date-math parsing.
* **Application API**: Deployed live on Render.

---

## 🛡️ Guardrails & Safety Architecture

### 1. Deterministic Out-of-Scope Filter
If a user requests tasks unrelated to the spa (e.g. *"Write a Python script"* or *"What is 15 * 30?"*):
* It is intercepted immediately at the `classify_intent` stage.
* The graph redirects to `handle_out_of_scope`.
* The server bypasses vector databases, tools, and heavy LLM chains entirely, returning a cached refusal message, avoiding unnecessary API charges.

### 2. State Checkpointing & Persistence
Using LangGraph's `PostgresSaver` checkpointing, the full conversation state is persisted automatically across the stateful graph:
* Every user-agent interaction step is versioned and saved using a `thread_id` (corresponds to the SQL database `Conversation.id`).
* This setup provides long-term application memory, fault tolerance for API timeouts, and unlocks the potential for "Time Travel" debugging or "Human-in-the-Loop" pausing.

### 3. Tool Binding Isolation
We maintain separate Tool Nodes (`booking_tool_node` and `reschedule_tool_node`) for different phases of the graph. This limits the agent's attack surface, ensuring rescheduling tools cannot accidentally run during booking phases.

---

## 📂 Project Structure (Backend)

```bash
back-end/
├── app/
│   ├── agent/
│   │   ├── graph.py       # LangGraph compilation, node declaration, and edges
│   │   ├── nodes.py       # Node implementation logic and prompts
│   │   ├── state.py       # AgentState TypedDict schema
│   │   ├── tools.py       # Database-bound tools (booking, modify slots)
│   │   └── chat_model.py  # Model initialization interfaces (Groq, Gemini)
│   ├── api/
│   │   ├── routes/
│   │   │   └── chat.py    # Main POST /chat API endpoint
│   │   └── dependencies.py# DB injection helper
│   ├── core/
│   │   ├── config.py      # Pydantic Settings
│   │   ├── database.py    # Database connection engine
│   │   └── vectordb.py    # Pinecone vector store retrieval setup
│   └── models/
│       └── domain.py      # SQLModel database tables (User, Appointment, Conversation, Message)
├── .env                   # Environment variables (Neon, Groq, Gemini, Pinecone)
└── requirements.txt       # Dependencies
```

---

## ⚙️ Running Locally (Backend)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Environment**:
   Make sure you copy and configure `.env` with the necessary keys (Neon PostgreSQL link, Groq/Gemini Keys, Pinecone API settings).
3. **Start the API Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend API docs will be available at `http://localhost:8000/docs`.

---
---

## 2.  Frontend Dashboard (`front-end/`)

A luxury, clean visual interface for clients to chat with **Nova** and developers to inspect the state machine in real-time. Deployed on [Vercel](https://nova-desk-ai-agent.vercel.app).

## 🚀 Key Features

* **Real-time Dev Mode (State Inspector)**: A side-by-side debugging panel displaying the current conversation state, extracted entity variables, parsed intent, and executing tools on every message.
* **Latency Counter**: Displays response latency in milliseconds to monitor system and API performance.
* **Rich Markdown Support**: Fully parses Markdown text responses (including treatment lists, bold summaries, and line formatting) using `react-markdown`.
* **State Reset capability**: Single-click session wipe (`Reset Session`) to start fresh and reset checkpointer memory.
* **Interactive UI**: Elegant luxury spa design language, featuring smooth transitions, rounded cards, and dev tools toggle switches.

---

## 🛠️ Technology Stack (Frontend)

| Component | Technology | Description / Usage |
| :--- | :--- | :--- |
| **Framework** | [React 19](https://react.dev/) | Virtual DOM library for responsive UI layout |
| **Build Tool** | [Vite 8](https://vite.dev/) | Ultra-fast bundling, Hot Module Replacement (HMR), and server runner |
| **Markdown Parser**| [react-markdown](https://github.com/remarkjs/react-markdown) | Secure, native React component to render rich chat formats |
| **Oxlint** | [Oxlint](https://oxc.rs/docs/guide/usage/linter.html) | High performance JavaScript/JSX compiler and code-quality validator |
| **CSS styling** | Vanilla CSS | Fully customized tokens representing luxury spa styles, gradients, and custom components |

---

## 🌐 Hosting & Services (Frontend)

* **UI Hosting Platform**: Hosted on **Vercel** (`https://nova-desk-ai-agent.vercel.app`) with automatic deployment pipelines linked to GitHub.
* **API Integration**: Connects to the Render backend via `VITE_API_URL` configuration inside build-time environment variables.

---

## 📂 Project Structure (Frontend)

```bash
front-end/
├── public/                # Static public assets (icons, images)
├── src/
│   ├── assets/            # App branding (logos, graphics)
│   ├── components/        # Isolated UI sections:
│   │   ├── ChatPanel.jsx       # Chat window UI and message rendering
│   │   ├── InspectorPanel.jsx  # Live state tree visualizer for developers
│   │   └── JsonTree.jsx        # Recursive JSON structure viewer
│   ├── App.jsx            # Main app shell, local state manager, and API calls
│   ├── index.css          # Visual styles, color palettes, animations, and layouts
│   └── main.jsx           # Vite React mounting file
├── index.html             # Basic HTML entry point
├── package.json           # Scripts, metadata, and packages
└── vite.config.js         # Vite configuration with React compiler plugin
```

---

## ⚙️ Running Locally (Frontend)

1. **Install Dependencies**:
   ```bash
   npm install
   ```
2. **Setup Environment**:
   Create a `.env.local` file and add the local backend API server path:
   ```env
   VITE_API_URL=http://localhost:8080
   ```
3. **Start Dev Server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.
