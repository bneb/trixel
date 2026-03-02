/* tslint:disable */
/* eslint-disable */

/**
 * Decode a Trixel PNG image from raw bytes.
 *
 * Called from JavaScript:
 * ```js
 * const result = decode_png(new Uint8Array(pngBuffer), 10);
 * ```
 *
 * Returns the decoded string (e.g., a URL) or throws on error.
 */
export function decode_png(png_bytes: Uint8Array, module_size: number): string;

/**
 * Try multiple common module sizes and return the first successful decode.
 *
 * This is the "just scan it" entry point — the user doesn't need to know
 * what module size was used during encoding.
 */
export function decode_png_auto(png_bytes: Uint8Array): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly decode_png: (a: number, b: number, c: number) => [number, number, number, number];
    readonly decode_png_auto: (a: number, b: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
