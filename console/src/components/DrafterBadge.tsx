import { describeDrafter } from "../summary";

export function DrafterBadge({ label }: { label?: string }) {
  const d = describeDrafter(label);
  return <span className={`badge drafter-${d.kind}`} title={d.long}>{d.short}</span>;
}
