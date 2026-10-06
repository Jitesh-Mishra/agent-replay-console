import { createServer } from "node:http";
import { createGateway } from "./gateway.js";

const port = Number(process.env.PORT ?? 4000);
createServer(createGateway()).listen(port, () => {
  console.log(`gateway on http://localhost:${port}/graphql -> ${process.env.ARC_API ?? "http://127.0.0.1:8000"}`);
});
