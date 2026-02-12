export const metadata = {
  title: "CSRAI UI V2"
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body style={{ margin: 0, background: "#0b0d10", color: "white" }}>
        {children}
      </body>
    </html>
  );
}
