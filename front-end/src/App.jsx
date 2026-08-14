import { useState, useCallback } from 'react';
import ChatPanel from './components/ChatPanel';
import InspectorPanel from './components/InspectorPanel';
import logo from './assets/logo.png';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [showInspector, setShowInspector] = useState(true);
  const [latencyMs, setLatencyMs] = useState(0);
  const [agentState, setAgentState] = useState({
    intent: { user_intent: 'idle', confidence_score: '0.0' },
    appointment_details: {},
    reschedule_details: {},
    modification_details: {},
    executed_tools: [],
  });

  const sendMessage = useCallback(async (messageText) => {
    const text = messageText || input.trim();
    if (!text || isLoading) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const payload = {
        message: text,
      };
      if (conversationId) {
        payload.conversation_id = conversationId;
      }

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // Save conversation ID from backend
      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
        localStorage.setItem('conversation_id', data.conversation_id);
      }

      // Add AI message
      const aiMessage = { role: 'assistant', content: data.message };
      setMessages((prev) => [...prev, aiMessage]);

      // Update inspector state
      if (data.agent_state) {
        setAgentState(data.agent_state);
      }
      if (data.latency_ms) {
        setLatencyMs(data.latency_ms);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        role: 'assistant',
        content:
          "I'm sorry, I'm having trouble connecting to the server right now. Please try again in a moment.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, conversationId]);

  const handleSend = () => sendMessage();

  const resetSession = () => {
    setMessages([]);
    setConversationId(null);
    setInput('');
    setLatencyMs(0);
    setAgentState({
      intent: { user_intent: 'idle', confidence_score: '0.0' },
      appointment_details: {},
      reschedule_details: {},
      modification_details: {},
      executed_tools: [],
    });
    localStorage.removeItem('conversation_id');
  };

  return (
    <>
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <img src={logo} alt="Nova Desk" className="header-logo" />
          <div className="header-brand">
            <span className="header-title">NOVA DESK</span>
            <span className="header-subtitle">AI Med Spa Concierge</span>
          </div>
        </div>

        <div className="header-right">


          <button className="btn-reset" onClick={resetSession}>
            Reset Session
          </button>

          <div className="toggle-wrapper">
            <span className="toggle-label">Dev Mode</span>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={showInspector}
                onChange={(e) => setShowInspector(e.target.checked)}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <ChatPanel
          messages={messages}
          isLoading={isLoading}
          input={input}
          setInput={setInput}
          onSend={handleSend}
        />

        {showInspector && (
          <InspectorPanel agentState={agentState} latencyMs={latencyMs} />
        )}
      </main>
    </>
  );
}

export default App;
