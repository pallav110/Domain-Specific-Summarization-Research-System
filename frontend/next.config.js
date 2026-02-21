/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  turbopack: {
    resolveAlias: {
      canvas: { browser: './empty-module.js' },
    },
  },
}

module.exports = nextConfig
