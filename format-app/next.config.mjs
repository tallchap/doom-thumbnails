import { execSync } from "child_process";

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_GIT_SHA: execSync("git rev-parse --short HEAD").toString().trim(),
  },
};

export default nextConfig;
