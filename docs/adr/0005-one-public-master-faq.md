# Answer clinic policy questions from one public master FAQ

The clinic maintains one master FAQ whose entire contents are safe to share with patients and customers. A clinic admin supplies the whole document. Editorial approval workflows are deferred.

An upload replaces the active document once the new file is fully indexed and searchable. If replacement fails, the previous file continues to serve answers. Ingestion derives the internal document version and entry identifiers from the file. Retrieval and Redis use that version so the next answer, including in an ongoing conversation, uses the current file.

Only public FAQ text and its source references are eligible for shared caching. Keep patient questions, booking details and personalized replies in their own conversation. When the master FAQ lacks an answer, Nova offers a human handoff.
