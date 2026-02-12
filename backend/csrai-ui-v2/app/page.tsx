"use client";

import { useState } from "react";

export default function Page() {
  const [files, setFiles] = useState<FileList | null>(null);

  return (
    <main style={{ padding: 40 }}>
      <h1>CSRAI MVP V2</h1>

      <input
        type="file"
        multiple
        onChange={(e) => setFiles(e.target.files)}
      />

      <div style={{ marginTop: 20 }}>
        선택된 파일 수: {files ? files.length : 0}
      </div>
    </main>
  );
}
