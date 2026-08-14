import { useState } from 'react';

/**
 * JsonTree — A recursive, interactive JSON tree viewer for the Inspector panel.
 * Renders keys in gold, strings in green, numbers in blue, booleans in orange, null in red.
 * Object/Array nodes are collapsible.
 */
function JsonTree({ data, name = null, defaultExpanded = true }) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  if (data === null || data === undefined) {
    return (
      <div className="json-line">
        {name && (
          <>
            <span className="json-key">{name}</span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-null">null</span>
      </div>
    );
  }

  if (typeof data === 'string') {
    return (
      <div className="json-line">
        {name && (
          <>
            <span className="json-key">{name}</span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-string">"{data}"</span>
      </div>
    );
  }

  if (typeof data === 'number') {
    return (
      <div className="json-line">
        {name && (
          <>
            <span className="json-key">{name}</span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-number">{data}</span>
      </div>
    );
  }

  if (typeof data === 'boolean') {
    return (
      <div className="json-line">
        {name && (
          <>
            <span className="json-key">{name}</span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-boolean">{data.toString()}</span>
      </div>
    );
  }

  // Arrays and Objects
  const isArray = Array.isArray(data);
  const entries = isArray ? data.map((v, i) => [i, v]) : Object.entries(data);
  const openBracket = isArray ? '[' : '{';
  const closeBracket = isArray ? ']' : '}';
  const isEmpty = entries.length === 0;

  if (isEmpty) {
    return (
      <div className="json-line">
        {name && (
          <>
            <span className="json-key">{name}</span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-bracket">{openBracket}{closeBracket}</span>
      </div>
    );
  }

  return (
    <div>
      <div className="json-line">
        <span
          className="json-toggle"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? '▼' : '▶'}
        </span>
        {name && (
          <>
            <span className="json-key" onClick={() => setExpanded(!expanded)}>
              {name}
            </span>
            <span className="json-colon">:</span>
          </>
        )}
        <span className="json-bracket">{openBracket}</span>
        {!expanded && (
          <span className="json-bracket"> … {closeBracket}</span>
        )}
      </div>
      {expanded && (
        <div className="json-node">
          {entries.map(([key, value]) => (
            <JsonTree
              key={key}
              data={value}
              name={isArray ? key : key}
              defaultExpanded={false}
            />
          ))}
          <div className="json-line">
            <span className="json-bracket">{closeBracket}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default JsonTree;
