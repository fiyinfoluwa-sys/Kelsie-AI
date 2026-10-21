"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { notifyExtension } from "@/lib/extension";

/**
 * Nudges the extension (if installed) to re-check the account whenever a
 * Kelsie page loads, e.g. right after signing out or changing sites. The
 * message carries no data: the extension always fetches the truth from
 * the server itself.
 */
export default function ExtensionSync() {
  const pathname = usePathname();
  useEffect(() => {
    notifyExtension();
  }, [pathname]);
  return null;
}
