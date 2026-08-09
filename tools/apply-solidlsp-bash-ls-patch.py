#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply-solidlsp-bash-ls-patch.py

Re-applies the solidlsp `bash_language_server.py` patch after a solidlsp upgrade.

Why this exists:
  solidlsp (a dependency of Serena) bundles bash-language-server 5.6.0, which
  deprecated its environment-variable configuration (GLOB_PATTERN, SHELLCHECK_PATH, ...)
  in favour of workspace configuration. On top of that, the default globPattern
  `**/*@(.sh|.inc|.bash|.command)` caused bash-language-server to parse C/C++ `.inc`
  files (e.g. `r8brain-free-src/*.inc`) as shell, producing spurious syntax-error
  warnings.

  This patch fixes both by delivering a `bashIde` configuration section
  (globPattern without `.inc`, shellcheckPath) through the proper LSP
  `workspace/configuration` request handler and a `workspace/didChangeConfiguration`
  notification, and by removing the deprecated env vars from the launched process.

  Because the patch lives inside site-packages, a `pip install --upgrade solidlsp`
  (or `uv tool upgrade serena-agent`) silently overwrites it. Run this script
  to re-apply it.

Idempotent: safe to run repeatedly; already-patched files are skipped.

Usage:
  python tools/apply-solidlsp-bash-ls-patch.py [--check] [--file PATH ...]

  --check       only report status, do not modify anything
  --file PATH   add an extra target file (can be repeated)

Exit codes:
  0  every target file is patched (or nothing to do)
  1  error (missing file, syntax error, invalid rule)
  2  at least one target file still needs patching
"""

import ast
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Default target files (in priority order). The uv-tool copy is the one actually
# used by Serena; the pip copy is kept in sync as a fallback.
# ---------------------------------------------------------------------------
DEFAULT_TARGETS = [
    os.path.join(
        os.path.expanduser("~"),
        "AppData",
        "Roaming",
        "uv",
        "tools",
        "serena-agent",
        "Lib",
        "site-packages",
        "solidlsp",
        "language_servers",
        "bash_language_server.py",
    ),
    os.path.join(
        os.path.expanduser("~"),
        "AppData",
        "Roaming",
        "Python",
        "Python314",
        "site-packages",
        "solidlsp",
        "language_servers",
        "bash_language_server.py",
    ),
]

# ---------------------------------------------------------------------------
# Replacement rules. Each rule transforms one contiguous block of the ORIGINAL
# (unpatched) file into the PATCHED text.
# ---------------------------------------------------------------------------
REPLACEMENTS = [
    {
        "name": "add DEFAULT_BASH_GLOB_PATTERN constant",
        "old": '''_SHELLCHECK_ALLOWED_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)

# Per-platform archive metadata:''',
        "new": '''_SHELLCHECK_ALLOWED_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)

# Default glob pattern used by bash-language-server's background analysis to find shell scripts.
# `.inc` is deliberately excluded here: it is commonly used for C/C++ include files (e.g. in
# r8brain-free-src), which are not shell and would otherwise trigger spurious parse errors.
# This can be overridden per-project via Serena's `ls_specific_settings` (key `bash.globPattern`).
DEFAULT_BASH_GLOB_PATTERN = "**/*@(.sh|.bash|.command)"
# Per-platform archive metadata:''',
    },
    {
        "name": "add _managed_shellcheck_path + _build_bashide_config helpers",
        "old": '''    if os.name == "nt":
        return os.path.join(install_dir, "shellcheck.exe")
    return os.path.join(install_dir, f"shellcheck-v{_SHELLCHECK_VERSION}", "shellcheck")


class BashLanguageServer(SolidLanguageServer):''',
        "new": '''    if os.name == "nt":
        return os.path.join(install_dir, "shellcheck.exe")
    return os.path.join(install_dir, f"shellcheck-v{_SHELLCHECK_VERSION}", "shellcheck")


def _managed_shellcheck_path(ls: "BashLanguageServer") -> str:
    """
    Returns the absolute path to the ShellCheck binary managed by SolidLSP, or "" if unknown.
    """
    bash_language_server_version = ls.custom_settings.get("bash_language_server_version", DEFAULT_BASH_LANGUAGE_SERVER_VERSION)
    ls_dirname = (
        "bash-lsp"
        if bash_language_server_version == INITIAL_BASH_LANGUAGE_SERVER_VERSION
        else f"bash-lsp-{bash_language_server_version}"
    )
    bash_ls_dir = os.path.join(ls._ls_resources_dir, ls_dirname)
    binary_path = _shellcheck_binary_path(bash_ls_dir)
    return binary_path if os.path.exists(binary_path) else ""


def _build_bashide_config(ls: "BashLanguageServer") -> dict:
    """
    Builds the ``bashIde`` workspace configuration section delivered to bash-language-server.

    bash-language-server 5.6.0 deprecates its environment-variable configuration
    (GLOB_PATTERN, SHELLCHECK_PATH, ...) in favour of workspace configuration. This helper
    assembles that section from the language server's custom settings (Serena's
    ``ls_specific_settings.bash``), falling back to built-in defaults, so the LS no longer
    needs to read those variables from the process environment.
    """
    config: dict = {}
    glob_pattern = ls.custom_settings.get("globPattern")
    if glob_pattern:
        config["globPattern"] = glob_pattern
    else:
        config["globPattern"] = DEFAULT_BASH_GLOB_PATTERN

    shellcheck_path = _managed_shellcheck_path(ls)
    if shellcheck_path:
        config["shellcheckPath"] = shellcheck_path
    return config


class BashLanguageServer(SolidLanguageServer):''',
    },
    {
        "name": "strip deprecated env vars from create_launch_command_env",
        "old": '''        def create_launch_command_env(self) -> dict[str, str]:
            bash_language_server_version = self._custom_settings.get("bash_language_server_version", DEFAULT_BASH_LANGUAGE_SERVER_VERSION)
            bash_ls_dir = self._resolve_bash_ls_dir(bash_language_server_version)
            managed_bin_dir = os.path.join(bash_ls_dir, "node_modules", ".bin")
            return {
                "PATH": managed_bin_dir + os.pathsep + os.environ.get("PATH", ""),
                "SHELLCHECK_PATH": _shellcheck_binary_path(bash_ls_dir),
            }''',
        "new": '''        def create_launch_command_env(self) -> dict[str, str]:
            # NOTE: bash-language-server 5.6.0 は環境変数設定を非推奨化している。
            # 親プロセスから継承されうる非推奨環境変数を除去してから子プロセス環境を構築する
            # （os.environ から除去しておけば、subprocess が os.environ.copy() で作る子環境にも
            #  渡らない）。設定は workspace/configuration (bashIde セクション) 経由で配信する。
            for _deprecated_env in (
                "GLOB_PATTERN",
                "SHELLCHECK_PATH",
                "INCLUDE_ALL_WORKSPACE_SYMBOLS",
                "BACKGROUND_ANALYSIS_MAX_FILES",
                "SHELLCHECK_ARGUMENTS",
                "EXPLAINSHELL_ENDPOINT",
                "SHFMT_PATH",
                "BASH_IDE_LOG_LEVEL",
            ):
                os.environ.pop(_deprecated_env, None)
            bash_language_server_version = self._custom_settings.get("bash_language_server_version", DEFAULT_BASH_LANGUAGE_SERVER_VERSION)
            bash_ls_dir = self._resolve_bash_ls_dir(bash_language_server_version)
            managed_bin_dir = os.path.join(bash_ls_dir, "node_modules", ".bin")
            return {
                "PATH": managed_bin_dir + os.pathsep + os.environ.get("PATH", ""),
            }''',
    },
    {
        "name": "advertise workspace.configuration capability",
        "old": '''                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "symbol": {"dynamicRegistration": True},
                },''',
        "new": '''                "workspace": {
                    "workspaceFolders": True,
                    "configuration": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "symbol": {"dynamicRegistration": True},
                },''',
    },
    {
        "name": "add workspace_configuration_handler",
        "old": '''        def execute_client_command_handler(params: dict) -> list:
            return []

        def do_nothing(params: dict) -> None:
            return''',
        "new": '''        def execute_client_command_handler(params: dict) -> list:
            return []

        def workspace_configuration_handler(params: dict) -> list:
            """
            Responds to ``workspace/configuration`` requests from bash-language-server,
            delivering the ``bashIde`` configuration section (globPattern, shellcheckPath)
            so the server does not have to rely on deprecated environment variables.
            """
            items = params.get("items", []) if isinstance(params, dict) else []
            result: list = []
            for item in items:
                section = item.get("section", "") if isinstance(item, dict) else ""
                if section == "bashIde":
                    result.append(_build_bashide_config(self))
                else:
                    result.append(None)
            return result

        def do_nothing(params: dict) -> None:
            return''',
    },
    {
        "name": "register workspace/configuration request handler",
        "old": '''        self.server.on_request("workspace/executeClientCommand", execute_client_command_handler)
        self.server.on_notification("$/progress", do_nothing)''',
        "new": '''        self.server.on_request("workspace/executeClientCommand", execute_client_command_handler)
        self.server.on_request("workspace/configuration", workspace_configuration_handler)
        self.server.on_notification("$/progress", do_nothing)''',
    },
    {
        "name": "push bashIde config after initialized",
        "old": '''        self.server.notify.initialized({})

        # Wait for server readiness with timeout''',
        "new": '''        self.server.notify.initialized({})

        # Deliver the bashIde configuration section via workspace/didChangeConfiguration so that
        # bash-language-server applies globPattern / shellcheckPath without relying on the
        # deprecated environment-variable configuration mechanism.
        bashide_config = _build_bashide_config(self)
        if bashide_config:
            self.server.notify.workspace_did_change_configuration({"settings": {"bashIde": bashide_config}})

        # Wait for server readiness with timeout''',
    },
]


def _is_patched(content: str) -> bool:
    """True if the file already contains the core markers of the patch."""
    return (
        "DEFAULT_BASH_GLOB_PATTERN" in content
        and "workspace_configuration_handler" in content
        and 'on_request("workspace/configuration"' in content
    )


def _validate_syntax(path: str, content: str) -> bool:
    try:
        ast.parse(content, filename=os.path.basename(path))
        return True
    except SyntaxError as exc:
        print(f"  [ERROR] syntax error in {path}: {exc}", file=sys.stderr)
        return False


def _apply_rules(content: str) -> tuple[str, list[str], list[str]]:
    """
    Applies all replacement rules to `content`.
    Returns (new_content, applied_rule_names, problematic_rule_names).
    """
    new_content = content
    applied: list[str] = []
    problems: list[str] = []
    for rule in REPLACEMENTS:
        name = rule["name"]
        old = rule["old"]
        new = rule["new"]
        if new in new_content:
            # already applied
            continue
        if old not in new_content:
            problems.append(name)
            continue
        count = new_content.count(old)
        if count != 1:
            problems.append(f"{name} (old block matched {count} times)")
            continue
        new_content = new_content.replace(old, new, 1)
        applied.append(name)
    return new_content, applied, problems


def process_file(path: str, check_only: bool) -> tuple[bool, str]:
    """
    Processes one target file.
    Returns (ok, status_message).
    """
    if not os.path.isfile(path):
        print(f"  [SKIP] not found: {path}")
        return True, "missing"

    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    if _is_patched(content):
        print(f"  [OK] already patched: {path}")
        return True, "patched"

    new_content, applied, problems = _apply_rules(content)

    if problems:
        print(f"  [WARN] some rules could not be applied to {path}:")
        for name in problems:
            print(f"    - {name}")
        # If nothing could be applied, leave the file untouched.
        if not applied:
            return False, "unpatched"

    if check_only:
        print(f"  [DRY-RUN] would patch {path} ({len(applied)} rule(s))")
        return False, "needs-patch"

    if new_content == content:
        print(f"  [SKIP] no changes needed: {path}")
        return True, "patched"

    # Validate syntax before writing.
    if not _validate_syntax(path, new_content):
        return False, "syntax-error"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_content)

    print(f"  [PATCHED] {path} ({len(applied)} rule(s))")
    return True, "patched"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-apply the solidlsp bash_language_server.py patch after upgrades."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report status, do not modify anything",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="add an extra target file (can be repeated)",
    )
    args = parser.parse_args()

    targets = DEFAULT_TARGETS + args.file

    print("solidlsp bash-language-server patch re-applier")
    print("mode: " + ("CHECK (no changes)" if args.check else "APPLY"))
    print()

    all_ok = True
    any_unpatched = False
    for path in targets:
        ok, status = process_file(path, args.check)
        if not ok:
            all_ok = False
        if status == "unpatched" or status == "needs-patch":
            any_unpatched = True

    print()
    if any_unpatched:
        print("Some target files still need patching (see messages above).")
        return 2
    if all_ok:
        print("All target files are patched.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
