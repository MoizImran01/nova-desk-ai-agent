import JsonTree from './JsonTree';

/**
 * InspectorPanel — Frosted-glass AI Engineer Inspector panel.
 * Displays real-time LangGraph agent state: intent classification,
 * tool execution stream, and live state JSON tree.
 */
function InspectorPanel({ agentState, latencyMs }) {
  const intent = agentState?.intent || {};
  const executedTools = agentState?.executed_tools || [];
  const appointmentDetails = agentState?.appointment_details || {};
  const rescheduleDetails = agentState?.reschedule_details || {};
  const modificationDetails = agentState?.modification_details || {};

  const confidenceScore = parseFloat(intent.confidence_score || '0');
  const confidenceClass =
    confidenceScore >= 0.8
      ? 'confidence-high'
      : confidenceScore >= 0.5
        ? 'confidence-medium'
        : 'confidence-low';

  const intentDisplay = intent.user_intent || 'idle';

  // Build combined state for the JSON viewer
  const hasAppointment = Object.keys(appointmentDetails).length > 0;
  const hasReschedule = Object.keys(rescheduleDetails).length > 0;
  const hasModification = Object.keys(modificationDetails).length > 0;

  return (
    <div className="inspector-panel">
      {/* Header */}
      <div className="inspector-header">
        <span className="inspector-title">Agent Inspector</span>
        {latencyMs > 0 && (
          <span className="latency-badge">{latencyMs}ms</span>
        )}
      </div>

      {/* Body */}
      <div className="inspector-body">
        {/* Intent & Routing */}
        <div className="inspector-card">
          <div className="inspector-card-header">
            <span className="inspector-card-title">Intent & Routing</span>
          </div>
          <div className="inspector-card-body">
            <div className="intent-row">
              <span className="intent-label">Classified Intent</span>
              <span className="intent-value">{intentDisplay}</span>
            </div>
            <div className="intent-row">
              <span className="intent-label">Confidence</span>
              <span className={`confidence-badge ${confidenceClass}`}>
                {confidenceScore.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Tool Call Stream */}
        <div className="inspector-card">
          <div className="inspector-card-header">
            <span className="inspector-card-title">Tool Calls</span>
          </div>
          <div className="inspector-card-body">
            {executedTools.length > 0 ? (
              <div className="tool-list">
                {executedTools.map((toolName, i) => (
                  <div className="tool-item" key={i} style={{ animationDelay: `${i * 0.1}s` }}>
                    <span className="tool-icon success">✓</span>
                    <span>{toolName}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="tool-empty">No tools executed this turn</p>
            )}
          </div>
        </div>

        {/* Live Agent State */}
        <div className="inspector-card">
          <div className="inspector-card-header">
            <span className="inspector-card-title">Live Agent State</span>
          </div>
          <div className="inspector-card-body">
            <div className="json-tree">
              {hasAppointment ? (
                <JsonTree
                  data={appointmentDetails}
                  name="appointment_details"
                  defaultExpanded={true}
                />
              ) : null}

              {hasReschedule ? (
                <JsonTree
                  data={rescheduleDetails}
                  name="reschedule_details"
                  defaultExpanded={true}
                />
              ) : null}

              {hasModification ? (
                <JsonTree
                  data={modificationDetails}
                  name="modification_details"
                  defaultExpanded={true}
                />
              ) : null}

              {!hasAppointment && !hasReschedule && !hasModification && (
                <p className="json-empty-state">No active state — waiting for user input</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default InspectorPanel;
