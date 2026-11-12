#include <windows.h>
#include <wchar.h>

static int launch_error(const wchar_t *message)
{
    MessageBoxW(NULL, message, L"YOLO Studio", MB_OK | MB_ICONERROR);
    return 1;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show)
{
    wchar_t root[32768];
    wchar_t python[32768];
    wchar_t command[32768];
    STARTUPINFOW startup = {sizeof(startup)};
    PROCESS_INFORMATION process = {0};
    DWORD exit_code = 1;

    DWORD length = GetModuleFileNameW(NULL, root, ARRAYSIZE(root));
    if (!length || length >= ARRAYSIZE(root)) {
        return launch_error(L"无法定位 YOLO Studio 的程序目录。");
    }
    wchar_t *separator = wcsrchr(root, L'\\');
    if (!separator) {
        return launch_error(L"无法定位 YOLO Studio 的程序目录。");
    }
    *separator = L'\0';

    if (swprintf(python, ARRAYSIZE(python), L"%ls\\runtime\\pythonw.exe", root) < 0 ||
        swprintf(command, ARRAYSIZE(command), L"\"%ls\" -I -B -X utf8 runtime\\launch.py", python) < 0) {
        return launch_error(L"程序目录路径过长。请解压到较短的路径后重试。");
    }
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, root, &startup, &process)) {
        return launch_error(L"无法启动内置 Python。请先完整解压 ZIP，再打开 YOLO Studio.exe。");
    }

    CloseHandle(process.hThread);
    if (WaitForSingleObject(process.hProcess, INFINITE) == WAIT_OBJECT_0) {
        GetExitCodeProcess(process.hProcess, &exit_code);
    }
    CloseHandle(process.hProcess);
    if (exit_code != 0) {
        return launch_error(L"YOLO Studio 异常退出。请查看程序目录中的 launch.log。\n"
                            L"请确认 ZIP 已完整解压，且当前用户可以写入该目录。");
    }
    return 0;
}
