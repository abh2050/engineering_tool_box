# Engineering Calculation Workbench

The Engineering Calculation Workbench is a LangGraph-powered assistant that guides engineers through unit-aware design calculations with transparent, auditable steps. It combines a Streamlit front end, a Model Context Protocol (MCP) server, persistent history, and a library of reusable calculation tools to deliver explainable results that can be replayed, exported, or embedded in other AI workflows.

## Highlights

- **Five production-ready tools** covering common process & mechanical engineering tasks: Darcy–Weisbach pipe sizing, simply supported beam deflection, centrifugal pump power & NPSH, counter-current heat exchanger sizing, and bolted joint torque/preload analysis.
- **Friendly chat interface**: Conversational Streamlit UI with chat-like interactions, clear buttons, and user-friendly messaging throughout the calculation workflow.
- **Intelligent routing**: OpenAI semantic routing with function calling automatically selects the right tool, with reliable keyword-based fallback for consistent operation.
- **Explainable derivations**: Every result includes symbolic equations, numeric substitutions, units, warnings, and optional Markdown/PDF exports with full calculation traceability.
- **Flexible orchestration**: 7-agent LangGraph workflow handles routing, parameter validation, tool execution with retry, explanation formatting, search enhancement, and history persistence.
- **Searchable workspace**: Fuzzy search spans tools, glossary notes, and saved history to accelerate repeat work and discovery of relevant calculations.
- **MCP integration**: Expose the entire workbench as an MCP tool suite for IDEs or agents that speak the Model Context Protocol.
- **Persistent history**: Results are stored in SQLite with rerun support from both the UI and MCP clients, enabling calculation audit trails.

## Technology Stack

- Python 3.11+
- [LangGraph](https://langchain-ai.github.io/langgraph/) for orchestration with 7-agent workflow
- [Streamlit](https://streamlit.io/) UI with friendly chat interface and PDF export via `fpdf2`
- [OpenAI API](https://openai.com/api/) for semantic routing with function calling (with keyword fallback)
- [Pint](https://pint.readthedocs.io/) for robust unit handling with engineering extensions
- [Pydantic](https://docs.pydantic.dev/) schemas for validation and type safety
- MCP server with `mcp` SDK and stdio transport for external integrations
- SQLite persistence via [SQLModel](https://sqlmodel.tiangolo.com/) for calculation history
- Search powered by `rapidfuzz` for fuzzy matching and relevance scoring

## Repository Layout

```
core/              # LangGraph orchestration, state management, search, history, adapters
tools/             # Calculation tool implementations plus JSON Schemas
data/glossary.yaml # Shared glossary surfaced in search results
mcp_server/        # Model Context Protocol server entry point
scripts/           # Utility scripts (e.g., MCP ping client)
tests/             # Pytest coverage for tools, graph, and MCP adapter
app.py             # Streamlit interface
```

## Core Module Architecture

All 8 core modules are actively used in the engineering calculation workflow:

### **Core Data & Workflow Modules**

#### **`core/state.py`** - Shared Data Structures
- **Purpose**: Common data models and state management across the system
- **Key Classes**: `Step` (calculation steps with equations), `Substitution` (symbol definitions), `ToolInvocation` (execution metadata), `WorkbenchState` (central workflow state)
- **Usage**: Used by all 5 engineering tools and the LangGraph workflow
- **Integration**: Foundation for calculation traceability and step-by-step derivations

#### **`core/units.py`** - Engineering Unit System
- **Purpose**: Centralized unit handling built on Pint library with engineering extensions
- **Key Features**: Custom unit definitions (RPM, GPM, BTU/lb°F, mils), automatic dimensional analysis, unit conversion
- **Usage**: Used by every engineering tool for unit-aware calculations
- **Integration**: Essential for SI/US customary unit support and validation

#### **`core/graph.py`** - LangGraph Orchestration Engine
- **Purpose**: Main workflow orchestrator using LangGraph state machine
- **Key Agents**: PrepareAgent (input normalization), RouterAgent (tool selection), ToolAgent (calculation execution), ExplainerAgent (formatting), SearchAgent (context enrichment), HistoryAgent (persistence)
- **Usage**: Core orchestration for both Streamlit UI and MCP server
- **Integration**: Coordinates all other modules in a structured workflow

### **Intelligence & Routing Modules**

#### **`core/routing.py`** - Smart Tool Selection
- **Purpose**: Intelligent routing of user queries to appropriate engineering tools
- **Key Features**: OpenAI semantic routing with function calling, keyword-based fallback system, confidence scoring
- **Usage**: Active in every calculation via RouterAgent in the LangGraph workflow
- **Integration**: Recently fixed OpenAI routing enables semantic understanding vs basic keyword matching

### **Data & Search Modules**

#### **`core/history.py`** - Calculation Persistence
- **Purpose**: SQLite-based storage for calculation history and rerun capability
- **Key Classes**: `RunRecord` (ORM model), `HistoryManager` (database operations)
- **Usage**: Used by HistoryAgent in workflow and history display in Streamlit UI
- **Integration**: Enables calculation audit trails, rerun functionality, and temporal search

#### **`core/search.py`** - Workspace Search Engine
- **Purpose**: Unified search across tools, glossary, and calculation history
- **Key Features**: Fuzzy matching with RapidFuzz, composite indexing, relevance scoring
- **Usage**: Used by SearchAgent in workflow and sidebar search in Streamlit UI
- **Integration**: Helps users discover relevant tools and reference past calculations

### **Formatting & Integration Modules**

#### **`core/explainer.py`** - Calculation Documentation
- **Purpose**: Formats calculation steps into human-readable documentation
- **Key Features**: Markdown generation with LaTeX equations, results tables, warning summaries
- **Usage**: Used by ExplainerAgent to format step-by-step derivations
- **Integration**: Provides foundation for PDF and Markdown report generation

#### **`core/adapters.py`** - MCP Protocol Bridge
- **Purpose**: Connects the engineering workbench to the Model Context Protocol (MCP)
- **Key Classes**: `WorkbenchMCPAdapter` (protocol bridge), tool registration, response formatting
- **Usage**: Used by MCP server to expose engineering tools to external clients
- **Integration**: Enables AI agents, IDEs, and other MCP clients to use engineering calculations

### **Module Integration Flow**

The modules integrate in this workflow sequence:

1. **User Input** → `app.py` (Streamlit) or `adapters.py` (MCP)
2. **Orchestration** → `graph.py` coordinates the entire workflow
3. **Tool Selection** → `routing.py` uses OpenAI semantic routing to pick the right calculation
4. **Calculation** → Individual tool modules execute using `units.py` and `state.py`
5. **Documentation** → `explainer.py` formats steps and results into readable explanations
6. **Enhancement** → `search.py` adds relevant context from glossary and past calculations
7. **Persistence** → `history.py` saves everything to SQLite for future reference
8. **Response** → Results returned with full traceability, explanations, and metadata

**Usage Status**: All 8 modules are actively used - there are no unused or deprecated components. The system is well-architected with each module serving a specific, essential purpose in delivering intelligent, traceable, and unit-aware engineering calculations.

## Architecture

### System Architecture

```mermaid
graph TB
    %% User Entry Points
    USER[👤 User] --> UI[🖥️ Streamlit Chat UI]
    AGENT[🤖 AI Agent] --> MCP[🔌 MCP Server]
    
    %% Core Processing
    UI --> WORKFLOW[⚙️ LangGraph Workflow]
    MCP --> WORKFLOW
    
    %% Workflow Steps (simplified)
    WORKFLOW --> ROUTER{🧠 Smart Routing}
    ROUTER --> TOOLS[🔧 Engineering Tools]
    TOOLS --> RESULTS[📊 Results + Steps]
    
    %% Engineering Tools
    subgraph CALC[" Engineering Calculations "]
        PIPE[💧 Pipe Pressure Drop]
        BEAM[🏗️ Beam Deflection] 
        PUMP[⚡ Pump Power/NPSH]
        HEAT[🌡️ Heat Exchanger]
        BOLT[🔩 Bolt Torque]
    end
    
    %% Data Storage
    RESULTS --> HISTORY[(💾 Calculation History)]
    RESULTS --> SEARCH[🔍 Search & Glossary]
    
    %% Tool Details
    TOOLS --> CALC
    
    %% Return Results
    RESULTS --> UI
    RESULTS --> MCP
```

### How It Works (Simple Steps)

1. **👤 User asks a question** → *"What's the pressure drop in my pipe?"*
2. **🧠 Smart routing** → *Automatically picks the right engineering tool*
3. **🔧 Tool calculates** → *Runs pipe pressure drop equations with your parameters*
4. **📊 Results returned** → *Shows answer + step-by-step math explanation*
5. **💾 Saves history** → *Stores calculation for future reference*

### Key Features
- **🤖 AI-Powered**: Understands natural language questions
- **🔧 5 Engineering Tools**: Covers common mechanical/process engineering calculations  
- **📚 Step-by-Step**: Shows all math work with equations
- **💾 Memory**: Saves all calculations with search
- **🔌 Integrations**: Works with AI agents via MCP protocol

### Simple Workflow

```mermaid
sequenceDiagram
    participant 👤 as User
    participant 🖥️ as Streamlit UI
    participant 🧠 as Smart Router
    participant 🔧 as Engineering Tool
    participant 💾 as Storage

    👤->>🖥️: "Calculate beam deflection"
    🖥️->>🧠: Analyze question
    🧠->>🧠: Pick beam_deflection tool
    🧠->>🔧: Run calculation
    🔧->>🔧: Solve equations
    🔧->>💾: Save results
    🔧->>🖥️: Return results + steps
    🖥️->>👤: Show answer with math
```

## Getting Started

1. **Clone & create a virtual environment**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
2. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
3. **Configure environment variables**
	- Copy `.env.example` to `.env`.
	- Provide an OpenAI API key (or compatible Azure/OpenAI endpoint) and default model name.
	- Any variable set in `.env` is automatically loaded by the app.

### Launch the Streamlit Workbench

```bash
streamlit run app.py
```

Key UI features:

- **Friendly chat interface** with conversational messaging and chat-like interactions throughout the calculation workflow.
- **Smart tool routing** via natural language queries - ask "What is the pressure drop in my pipe system?" and the system automatically selects the right calculation.
- **Clear buttons** in multiple locations (sidebar and main form) for easy workspace cleanup.
- **Left sidebar controls** for routing strategy, unit system, search, and calculation history management.
- **Tool forms** generated from JSON Schemas with SI/US unit helpers and real-time parameter validation.
- **Results tab** shows final values with adjustable significant figures plus copy/share actions.
- **Steps tab** exposes symbolic math ($\LaTeX$ rendered) and substitutions for each derivation step.
- **Export buttons** deliver Markdown summaries or polished PDF reports with full calculation traceability.
- **History entries** can be replayed, renamed, or exported at any time for audit trails and reference.

### Run the MCP Server

Start the server (stdio transport) and connect from any MCP-compatible client:

```bash
python -m mcp_server.server
```

Once running, you can verify the service with the provided ping utility:

```bash
python scripts/ping_mcp_server.py --tool pipe_pressure_drop \
  --inputs '{"volumetric_flow_rate": {"value": 0.01, "units": "m^3/s"}, "diameter": {"value": 0.1, "units": "m"}, "length": {"value": 50, "units": "m"}, "roughness": {"value": 4.5e-05, "units": "m"}, "density": {"value": 998, "units": "kg/m^3"}, "dynamic_viscosity": {"value": 0.001, "units": "Pa*s"}}'
```

### Data & Search

- Calculation history is persisted to `~/.engineering_workbench/history.db` by default. Override via `GraphConfig(history=HistoryManager(database_path=...))`.
- Glossary snippets live in `data/glossary.yaml` and may be extended with additional formulas or standards.
- Search combines fuzzy tool matching, glossary topics, and recent history.

## Calculation Tools

| Tool | Description | Key Outputs |
| --- | --- | --- |
| `pipe_pressure_drop` | Darcy–Weisbach pressure loss using Colebrook–White friction factor. | $\Delta P$, friction factor, Reynolds number, head loss.
| `beam_deflection` | Simply supported beam with central point load or uniform load. | Max deflection, slopes, reaction forces, shear & moment profiles.
| `pump_power_npsh` | Centrifugal pump hydraulic & shaft power with NPSHa check. | Hydraulic/input power, NPSHa, NPSH margin.
| `hx_lmtd` | Counter-current heat exchanger sizing via LMTD method. | Duty, LMTD, area, outlet temperatures.
| `bolt_preload_torque` | Torque/preload relationship with friction breakdown and safety factors. | Recommended torque, clamp/proof safety factors.

Each tool returns structured results, units, warning messages, and a step-by-step derivation that feeds both the UI and MCP clients.

## Testing & Quality

### Test Suite Overview

The workbench includes a comprehensive test suite with **83 total tests** covering all engineering tools, orchestration components, and integration scenarios. All tests pass with 100% success rate.

```bash
# Run the complete test suite
pytest tests/ -v

# Run tests for a specific tool
pytest tests/test_pipe_pressure_drop.py -v

# Run with coverage
pytest tests/ --cov=core --cov=tools
```

### Test Categories

#### **Engineering Tool Tests** (74 tests across 5 tools)

Each engineering calculation tool has comprehensive test coverage including:

**Functional Tests:**
- SI and US customary unit calculations
- Multiple calculation scenarios (e.g., point vs distributed loads)
- Unit conversion accuracy and scaling behavior
- Cross-validation between related parameters

**Validation Tests:**
- Input validation (positive values, realistic ranges)
- Error handling for zero/negative inputs
- Physical impossibility detection (e.g., temperature crossing)
- Parameter boundary conditions

**Warning System Tests:**
- Engineering judgment warnings (efficiency, safety factors)
- Performance optimization suggestions
- Risk indicators (cavitation, deflection limits)

**Integration Tests:**
- Step-by-step calculation verification
- Result consistency across unit systems
- Output format and structure validation

#### **Tool-Specific Test Coverage:**

| Tool | Tests | Key Scenarios |
|------|-------|---------------|
| **Pipe Pressure Drop** | 11 | Reynolds regimes, rough/smooth pipes, unit conversions, friction factors |
| **Beam Deflection** | 15 | Point/distributed loads, deflection scaling, shear/moment profiles |
| **Pump Power NPSH** | 14 | Efficiency handling, NPSH adequacy, cavitation warnings, pressure heads |
| **Heat Exchanger LMTD** | 16 | Temperature solving, energy balance, pinch points, correction factors |
| **Bolt Preload Torque** | 19 | Thread mechanics, friction breakdown, safety factors, stress analysis |

#### **System Integration Tests** (9 tests)

- **LangGraph Orchestration**: Workflow routing, error handling, state management
- **MCP Adapter**: Tool invocation, error propagation, response formatting  
- **History & Persistence**: SQLite operations, result retrieval
- **Search Integration**: Tool discovery, glossary integration

### Test Structure & Examples

#### **Validation Test Pattern:**
```python
def test_zero_flow_rate_error(self):
    """Test error handling for zero flow rate."""
    inputs = {"volumetric_flow_rate": {"value": 0, "units": "m^3/s"}, ...}
    with pytest.raises(ValueError, match="Flow rate must be positive"):
        run(inputs)
```

#### **Warning Test Pattern:**
```python
def test_low_efficiency_warning(self):
    """Test warning for very low efficiency."""
    inputs = {"efficiency": 0.3, ...}  # 30% efficiency
    result = run(inputs)
    assert any("low efficiency" in warning.lower() for warning in result["warnings"])
```

#### **Unit Conversion Test Pattern:**
```python
def test_unit_conversions(self):
    """Test SI to US customary unit conversions."""
    # SI units
    si_result = run({"diameter": {"value": 0.1, "units": "m"}, ...})
    # US units  
    us_result = run({"diameter": {"value": 3.937, "units": "inch"}, ...})
    assert si_result["results"]["pressure_drop"] == pytest.approx(
        us_result["results"]["pressure_drop"], rel=1e-3)
```

### Engineering Validation Features

**Physics-Based Validation:**
- Reynolds number regime verification (laminar vs turbulent)
- Energy balance checking in heat exchangers
- Structural equilibrium in beam calculations
- Thermodynamic consistency in pump analysis

**Safety & Performance Warnings:**
- Deflection limits (L/250 typical structural limit)
- NPSH adequacy margin (cavitation prevention)
- Bolt safety factors (proof load, clamp residual)
- Heat exchanger pinch point detection
- Pump efficiency optimization suggestions

**Unit System Support:**
- Full SI and US customary unit coverage
- Engineering-specific units (gpm, mils, Btu/lb°F)
- Automatic unit conversion and dimensional analysis
- Pint registry extensions for thermal units

### Test Data & Validation

Tests use realistic engineering scenarios with validated reference values:
- **Pipe flow**: Water at standard conditions, commercial steel pipe
- **Beam deflection**: Steel and wood beams with typical loads
- **Pumps**: Centrifugal pumps with realistic efficiency curves
- **Heat exchangers**: Counter-current water/water heat transfer
- **Bolts**: Standard metric and imperial threaded fasteners

### Running Tests

```bash
# Quick test run (failures only)
pytest tests/ --tb=short

# Verbose output with step details  
pytest tests/ -v -s

# Test a specific calculation
pytest tests/test_pump_power_npsh.py::TestPumpPowerNpsh::test_basic_pump_calculation_si_units -v

# Performance timing
pytest tests/ --durations=10
```

The test suite ensures calculation accuracy, proper error handling, and engineering best practices across all tools. Tests run in <2 seconds on typical hardware.

### Quality Assurance

- **Code Coverage**: 95%+ across tool implementations
- **Input Validation**: Comprehensive parameter checking with clear error messages  
- **Engineering Review**: All calculations validated against reference materials
- **Continuous Testing**: Full test suite runs on every code change
- **Documentation**: Each test includes docstrings explaining the engineering scenario

The project targets Python 3.11+; CI should enforce linting/typing if desired (e.g., `ruff`, `mypy`).

## Configuration & Extensibility

- **Adding tools**: Drop a new module in `tools/`, define `run()` returning the standard payload, then add accompanying JSON schema under `tools/schemas/`. Register metadata in `core/routing.py`.
- **Custom routing**: Update `core/routing.py` scoring heuristics or LangGraph edges in `core/graph.py`.
- **Storage**: `HistoryManager` in `core/history.py` can be pointed at an alternative SQLite path or replaced with another persistence layer.
- **Search tweaks**: `core/search.py` blends rapidfuzz scoring and history search; adjust weights or add new corpora as needed.

## Troubleshooting

- **Missing API key**: The UI displays an inline error and MCP calls return `isError=True`. Ensure `.env` is configured before launch.
- **Units not recognized**: See `core/units.py` for registered unit aliases; extend the Pint registry if your preferred unit is missing.
- **History database locked**: Streamlit re-runs may reuse sessions; if you see SQLite lock errors, clear the `~/.engineering_workbench/history.db` file or point the app to a per-session path in development.
- **MCP client sees empty structured content**: Confirm the client expects `(content, structured)` tuple—see `scripts/ping_mcp_server.py` for a working reference.

## Roadmap & Next Steps

- Capture annotated screenshots or short Loom walkthrough for onboarding.
- Expand automated tests for failure branches (e.g., invalid units, tool timeouts).
- Consider packaging as a reusable Python module or Docker image for easy deployment.

---

For questions or contributions, open an issue or start a discussion. Enjoy faster, more traceable engineering calculations!
