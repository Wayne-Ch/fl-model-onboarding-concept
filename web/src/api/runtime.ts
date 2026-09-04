export class ApiParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiParseError";
  }
}

export function asRecord(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ApiParseError(`Expected ${name} to be an object.`);
  }
  return value as Record<string, unknown>;
}

export function readUnknown(record: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }
  return undefined;
}

export function readString(record: Record<string, unknown>, keys: string[], fallback?: string): string {
  const value = readUnknown(record, keys);
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  if (fallback !== undefined) {
    return fallback;
  }
  throw new ApiParseError(`Expected string at ${keys.join(" | ")}.`);
}

export function readOptionalString(record: Record<string, unknown>, keys: string[]): string | undefined {
  const value = readUnknown(record, keys);
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (typeof value === "string") {
    return value;
  }
  throw new ApiParseError(`Expected optional string at ${keys.join(" | ")}.`);
}

export function readBoolean(record: Record<string, unknown>, keys: string[], fallback?: boolean): boolean {
  const value = readUnknown(record, keys);
  if (typeof value === "boolean") {
    return value;
  }
  if (fallback !== undefined) {
    return fallback;
  }
  throw new ApiParseError(`Expected boolean at ${keys.join(" | ")}.`);
}

export function readOptionalBoolean(record: Record<string, unknown>, keys: string[]): boolean | undefined {
  const value = readUnknown(record, keys);
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "boolean") {
    return value;
  }
  throw new ApiParseError(`Expected optional boolean at ${keys.join(" | ")}.`);
}

export function readNumber(record: Record<string, unknown>, keys: string[]): number | undefined {
  const value = readUnknown(record, keys);
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  throw new ApiParseError(`Expected numeric value at ${keys.join(" | ")}.`);
}

export function readStringArray(record: Record<string, unknown>, keys: string[]): string[] {
  const value = readUnknown(record, keys);
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new ApiParseError(`Expected string array at ${keys.join(" | ")}.`);
  }
  const strings = value.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
  return Array.from(new Set(strings));
}

export function readRecord(record: Record<string, unknown>, keys: string[]): Record<string, unknown> | undefined {
  const value = readUnknown(record, keys);
  if (value === undefined || value === null) {
    return undefined;
  }
  return asRecord(value, keys.join(" | "));
}
