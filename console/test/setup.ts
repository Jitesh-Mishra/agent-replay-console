import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);   // vitest globals are off, so testing-library does not unmount between tests on its own
