/**
 * WebSocket client for JARVIS server communication.
 */

export type MessageHandler = (msg: Record<string, unknown>) => void;

export interface JarvisSocket {
  /** False when the socket was not open and the message was dropped. */
  send(data: Record<string, unknown>): boolean;
  onMessage(handler: MessageHandler): void;
  close(): void;
  isConnected(): boolean;
}

export function createSocket(url: string): JarvisSocket {
  let ws: WebSocket | null = null;
  let handlers: MessageHandler[] = [];
  let reconnectDelay = 1000;
  let closed = false;
  let connected = false;

  function connect() {
    if (closed) return;

    ws = new WebSocket(url);

    ws.onopen = () => {
      connected = true;
      reconnectDelay = 1000;
      console.log("[ws] connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        for (const h of handlers) h(msg);
      } catch {
        console.warn("[ws] bad message", event.data);
      }
    };

    ws.onclose = () => {
      connected = false;
      if (!closed) {
        console.log(`[ws] reconnecting in ${reconnectDelay}ms`);
        setTimeout(connect, reconnectDelay);
        // Capped at 5s rather than 30s. A background tab has its timers
        // throttled hard by the browser, so a 30s delay can stretch to
        // minutes — which is how a page sat disconnected through three server
        // restarts while still looking alive, because the HUD polls over
        // plain HTTP and kept updating.
        reconnectDelay = Math.min(reconnectDelay * 2, 5000);
      }
    };

    ws.onerror = (err) => {
      console.error("[ws] error", err);
      ws?.close();
    };
  }

  connect();

  // Reconnect the moment the tab comes back to the foreground, without
  // waiting out whatever delay the throttled timer is sitting on.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !closed && !connected) {
      reconnectDelay = 1000;
      connect();
    }
  });

  return {
    send(data) {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
        return true;
      }
      // Dropping this silently is what made a dead socket look like a mute
      // JARVIS: the speech was recognised, composed, and thrown away with
      // nothing said about it. The caller decides how to tell the user.
      console.warn("[ws] not connected — message dropped", data);
      return false;
    },
    onMessage(handler) {
      handlers.push(handler);
    },
    close() {
      closed = true;
      ws?.close();
    },
    isConnected() {
      return connected;
    },
  };
}
