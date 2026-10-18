"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError, deleteCodebase, getCodebases, getHealth } from "./api";
import type { Codebase, Health } from "./types";

interface ServerState {
  health: Health | null;
  codebases: Codebase[] | null;
  error: string | null;
}

/** The server's status and its indexed codebases, refreshed after every change. */
export function useServer() {
  const [state, setState] = useState<ServerState>({ health: null, codebases: null, error: null });

  useEffect(() => {
    let active = true;
    void load().then((update) => {
      if (active) setState((current) => ({ ...current, ...update }));
    });
    return () => {
      active = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    const update = await load();
    setState((current) => ({ ...current, ...update }));
  }, []);

  const remove = useCallback(
    async (name: string) => {
      await deleteCodebase(name);
      await refresh();
    },
    [refresh],
  );

  return { ...state, refresh, remove };
}

/** The server's state, or the error that kept it from loading (the last state is kept). */
async function load(): Promise<Partial<ServerState>> {
  try {
    const [health, codebases] = await Promise.all([getHealth(), getCodebases()]);
    return { health, codebases, error: null };
  } catch (failure) {
    return {
      error: failure instanceof ApiError ? failure.message : "Could not reach the server.",
    };
  }
}
