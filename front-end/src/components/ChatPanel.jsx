import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import logo from '../assets/logo.png';
import ai_chat_bubble from '../assets/ai-chat-bubble-img.png';
/**
 * ChatPanel — The user-facing luxury chat interface.
 * Renders message history with markdown support, soft reveal replies,
 * typing indicator, and input bar.
 */

function ChatPanel({ messages, isLoading, input, setInput, onSend }) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const typingStatusRef = useRef(null);

  // Stable status while waiting for the AI reply
  if (isLoading && !typingStatusRef.current) {
    typingStatusRef.current = 'Thinking...';
  }
  if (!isLoading) {
    typingStatusRef.current = null;
  }

  // Auto-scroll while messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="chat-panel split">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && !isLoading ? (
          <div className="welcome-container">
            <img src={logo} alt="Nova Desk" className="welcome-logo" />
            <h2 className="welcome-title soft-reveal">Welcome to Nova Desk!</h2>
            <p className="welcome-text soft-reveal soft-reveal-delay">
              Your luxury AI concierge. How can I help you today?
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <div className="assistant-wrapper">
                    <img src={ai_chat_bubble} alt="Nova" className="ai-avatar" />
                    <div>
                      <div className="message-bubble assistant soft-reveal">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="message-bubble user">
                    {msg.content}
                  </div>
                )}
              </div>
            ))}
          </>
        )}

        {/* Typing Indicator */}
        {isLoading && (
          <div className="typing-indicator">
            <img src={logo} alt="Nova" className="ai-avatar" />
            <div className="typing-bubble">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span className="typing-status">{typingStatusRef.current}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            placeholder="Ask Nova Desk"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            className="btn-send"
            onClick={onSend}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatPanel;
