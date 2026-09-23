// Facts about a measurement that its payload does not carry.
//
// The /dash/api/results payload reports the `embed+rerank` retrieval arm as
// an ordinary row, but AGENTS.md and docs/FINDINGS.md #20 record that every
// reranker number in the repo is VOID: /v1/rerank scores depend on what else
// is in the same request, and search_code batches up to 40 candidates. A
// dashboard that printed 13.5% hit@1 without that would present a corrupted
// number as a result.
//
// This belongs in the payload (mcp/dash_results.py), not here. Until it is,
// the claim lives in one place with its source, so it can be deleted the
// moment the server says it itself.
export const KNOWN_VOID: Record<string, { why: string; source: string }> = {
  'embed+rerank': {
    why: 'VOID · scores depend on batch composition',
    source: 'docs/FINDINGS.md #20',
  },
};
