export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;

  return (
    <main style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700 }}>CSRAI UI (v2)</h1>
      <p style={{ marginTop: 8, opacity: 0.8 }}>
        Frontend is deployed. If you see this page, routing is OK.
      </p>

      <div style={{ marginTop: 16, padding: 16, border: "1px solid #ddd", borderRadius: 10 }}>
        <div style={{ fontWeight: 600 }}>API Base</div>
        <div style={{ marginTop: 6, fontFamily: "ui-monospace, SFMono-Regular" }}>
          {apiBase ? apiBase : "❌ NEXT_PUBLIC_API_BASE is not set"}
        </div>
        <p style={{ marginTop: 10, opacity: 0.8 }}>
          Set this in Vercel Environment Variables as:
          <br />
          <code>NEXT_PUBLIC_API_BASE = https://...runpod...proxy.runpod.net</code>
        </p>
      </div>
    </main>
  );
}
