import { proxy } from "@/lib/backend";

/** Indexing large batches and long polls can take a while. */
export const maxDuration = 300;

type Context = RouteContext<"/api/[...path]">;

/** `/api/*`: forwards the allowed routes to the backend (see `ROUTES`). */
async function handle(request: Request, context: Context): Promise<Response> {
  const { path } = await context.params;
  return proxy(request, path);
}

export function GET(request: Request, context: Context): Promise<Response> {
  return handle(request, context);
}

export function POST(request: Request, context: Context): Promise<Response> {
  return handle(request, context);
}

export function DELETE(request: Request, context: Context): Promise<Response> {
  return handle(request, context);
}
