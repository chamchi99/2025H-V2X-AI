// ============================================================
//  YOLO Data Receiver (C++ Client)
//  - Python AI 서버(:9000)에서 NDJSON(frame_update) 수신
//  - JSON을 파싱해 struct(TrafficObject/FrameData)에 저장  ← 다른 C++ 프로그램이 그대로 사용
//  - 최신 프레임을 대시보드 형태로 예쁘게 출력(화면 갱신, 터미널 폭주 없음)
//  빌드:  g++ -O2 -pthread -I. yolo_receiver.cpp -o yolo_receiver
//         (json.hpp 가 같은 폴더에 있어야 함)
// ============================================================
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <csignal>
#include <fcntl.h>

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "json.hpp"
using json = nlohmann::json;

static int           g_conn_sock = -1;
static volatile bool g_stop      = false;

// ---- 파싱된 데이터 구조체 (다른 C++ 프로그램이 이 형태로 값을 읽어 쓰면 됨) ----
struct TrafficObject {
    int    cctv          = 0;      // 카메라 번호 (1/2/3)
    int    cls           = 0;      // 클래스 0:person 1:cyclist 2:car 3:motorcycle 4:truck
    int    id            = 0;      // 트랙 ID (88888 = AV)
    double lat           = 0.0;    // 위도
    double lon           = 0.0;    // 경도
    double speed         = 0.0;    // km/h
    double heading       = 0.0;    // 진행방향 (radian)
    int    lane          = -1;     // 차선 (-1/0/1)
    int    stop_duration = 0;      // 정차 시간 (초)
    std::string ped_zone;          // 보행자 위치 (sidewalk/road/crosswalk 등)
    std::vector<double> lane_occupancy;  // [lane1, lane2] 차선별 점유율 (정차차만)
    double bbox[4]       = {0,0,0,0};    // x1,y1,x2,y2 (픽셀)
    std::vector<std::array<double,2>> bbox_gps;  // 회전 사각형 4모서리 GPS [[lat,lon],...] (지도 표출용)
};

struct FrameData {
    int         frame_idx        = -1;
    double      ai_latency       = 0.0;   // ms
    long long   frame_start_time = 0;
    long long   frame_end_time   = 0;
    std::vector<TrafficObject> objects;
};

// 다른 스레드/프로그램이 참조할 최신 데이터
FrameData   g_latest_frame;          // ← 파싱된 struct (권장)
std::string g_latest_frame_data = ""; // ← 원본 JSON 문자열 (필요시)

void signal_handler(int) { g_stop = true; std::cout << "\n[RX] Ctrl+C 감지. 종료합니다.\n"; }

// ---- payload(JSON 한 줄) → FrameData 파싱 ----
static bool parse_payload(const std::string& buf, FrameData& fd) {
    try {
        json p = json::parse(buf);
        if (!p.contains("cctv_traffic")) return false;
        const json& ct = p["cctv_traffic"];
        fd = FrameData{};
        fd.frame_idx        = ct.value("frame_idx", -1);
        fd.ai_latency       = ct.value("ai_latency", 0.0);
        fd.frame_start_time = ct.value("frame_start_time", 0LL);
        fd.frame_end_time   = ct.value("frame_end_time", 0LL);
        if (ct.contains("traffic") && ct["traffic"].is_array()) {
            for (const auto& t : ct["traffic"]) {
                TrafficObject o;
                o.cctv          = t.value("cctv", 0);
                o.cls           = t.value("class", 0);
                o.id            = t.value("id", 0);
                o.speed         = t.value("speed", 0.0);
                o.heading       = t.value("heading", 0.0);
                o.lane          = t.value("lane", -1);
                o.stop_duration = t.value("stop_duration", 0);
                // ped_zone은 보행자 아니면 null로 옴 → null이면 value()가 예외.
                // 문자열일 때만 취함(null/누락이면 빈 문자열 유지).
                if (t.contains("ped_zone") && t["ped_zone"].is_string())
                    o.ped_zone = t["ped_zone"].get<std::string>();
                if (t.contains("GPS") && t["GPS"].is_array() && t["GPS"].size() >= 2) {
                    o.lat = t["GPS"][0]; o.lon = t["GPS"][1];
                }
                if (t.contains("bbox") && t["bbox"].is_array() && t["bbox"].size() >= 4)
                    for (int i = 0; i < 4; ++i) o.bbox[i] = t["bbox"][i];
                if (t.contains("lane_occupancy") && t["lane_occupancy"].is_array())
                    o.lane_occupancy = t["lane_occupancy"].get<std::vector<double>>();
                if (t.contains("bbox_gps") && t["bbox_gps"].is_array())
                    for (const auto& pt : t["bbox_gps"])
                        if (pt.is_array() && pt.size() >= 2)
                            o.bbox_gps.push_back({ (double)pt[0], (double)pt[1] });
                fd.objects.push_back(o);
            }
        }
        return true;
    } catch (...) { return false; }
}

// ---- 최신 프레임을 대시보드로 출력 (화면 갱신 방식 → 폭주 없음) ----
static void print_dashboard(const FrameData& fd) {
    static const char* CN[] = {"person","cyclist","car","moto","truck"};
    const int AV_ID = 88888;

    printf("\033[H");  // 커서를 화면 좌상단으로 (덮어쓰기)
    printf("Frame %-7d  objects: %-3zu  AI latency: %6.1f ms\033[K\n\033[K\n",
           fd.frame_idx, fd.objects.size(), fd.ai_latency);
    printf("cam class    id     lat, lon              spd  head  lane  zone       stop/occ\033[K\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\033[K\n");
    for (const auto& o : fd.objects) {
        std::string cls   = (o.cls >= 0 && o.cls < 5) ? CN[o.cls] : "?";
        std::string idstr = (o.id == AV_ID) ? "AV" : std::to_string(o.id);
        double head_deg   = o.heading * 180.0 / M_PI;
        std::string zone  = o.ped_zone.empty() ? "-" : o.ped_zone;
        // stop / lane_occupancy
        std::string so;
        if (o.stop_duration > 0) so = std::to_string(o.stop_duration) + "s";
        for (size_t i = 0; i < o.lane_occupancy.size(); ++i)
            if (o.lane_occupancy[i] > 0)
                so += " L" + std::to_string(i+1) + ":" +
                      std::to_string((int)(o.lane_occupancy[i]*100)) + "%";
        if (so.empty()) so = "-";
        printf(" %2d %-7s %5s  %9.6f,%10.6f %5.1f %5.0f %4d  %-9s %-s\033[K\n",
               o.cctv, cls.c_str(), idstr.c_str(), o.lat, o.lon,
               o.speed, head_deg, o.lane, zone.c_str(), so.c_str());
        // bbox_gps 4모서리 (지도 표출용 회전 사각형) — 들여쓴 서브라인
        if (!o.bbox_gps.empty()) {
            printf("    └ bbox_gps:");
            size_t np = o.bbox_gps.size(); if (np > 4) np = 4;  // 닫힘점 제외 4모서리
            for (size_t i = 0; i < np; ++i)
                printf(" (%.5f,%.5f)", o.bbox_gps[i][0], o.bbox_gps[i][1]);
            printf("\033[K\n");
        }
    }
    printf("─────────────────────────────────────────────────────────────────────────────────\033[K\n");
    printf("\033[J");  // 커서 아래(이전 프레임 잔상) 지우기
    fflush(stdout);
}

// =========================================================
// 결과 수신 스레드
// =========================================================
void* result_rx_thread_fn(void*) {
    printf("[RESULT RX] 수신 스레드 시작됨\n");
    std::string buf;
    char c;
    while (!g_stop) {
        int n = recv(g_conn_sock, &c, 1, 0);
        if (n <= 0) {
            if (!g_stop) printf("\n[RESULT RX] AI 서버(Python)와 연결이 끊어졌습니다.\n");
            break;
        }
        if (c == '\r') continue;
        if (c != '\n') { buf += c; continue; }
        if (buf.empty()) continue;

        if (buf.find("frame_update") != std::string::npos) {
            g_latest_frame_data = buf;                 // 원본 보존
            FrameData fd;
            if (parse_payload(buf, fd)) {
                g_latest_frame = fd;                   // 파싱된 struct 갱신
                print_dashboard(fd);                   // 예쁘게 출력
            }
        }
        // (console_logs 페이로드는 현재 Python이 보내지 않음 → 처리 생략)
        buf.clear();
    }
    printf("[RESULT RX] 수신 스레드 종료됨\n");
    return nullptr;
}

// =========================================================
// TCP 연결 (무한 재시도)
// =========================================================
static bool connect_to_server(const char* host, int port) {
    g_conn_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_conn_sock < 0) { perror("[TCP] socket 생성 오류"); return false; }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    inet_pton(AF_INET, host, &addr.sin_addr);
    printf("[TCP] Python AI Server %s:%d 연결 시도...\n", host, port);
    while (!g_stop) {
        if (connect(g_conn_sock, (sockaddr*)&addr, sizeof(addr)) == 0) {
            printf("[TCP] 연결 성공! → %s:%d\n", host, port);
            return true;
        }
        printf("[TCP] 연결 실패. 2초 후 재시도...\n");
        sleep(2);
        close(g_conn_sock);
        g_conn_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (g_conn_sock < 0) { perror("[TCP] socket 생성 오류"); return false; }
    }
    return false;
}

// =========================================================
// Main
// =========================================================
int main(int argc, char* argv[]) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    const char* server_host = "127.0.0.1";
    int         server_port = 9000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--host" && i+1 < argc) server_host = argv[++i];
        else if (a == "--port" && i+1 < argc) server_port = std::stoi(argv[++i]);
    }

    printf("==============================================\n");
    printf("  YOLO Data Receiver (C++ Client)\n");
    printf("  TCP Server :  %s:%d   (q 입력시 종료)\n", server_host, server_port);
    printf("==============================================\n\n");

    int fd = fileno(stdin);
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    while (!g_stop) {
        if (!connect_to_server(server_host, server_port)) break;
        printf("\033[2J");  // 대시보드 시작 전 화면 클리어

        pthread_t t_result;
        pthread_create(&t_result, nullptr, result_rx_thread_fn, nullptr);

        while (!g_stop) {
            char buffer[256];
            int n = read(fd, buffer, sizeof(buffer) - 1);
            if (n > 0) {
                buffer[n] = '\0';
                std::string input(buffer);
                if (input.find("q") != std::string::npos) { g_stop = true; break; }
            }
            char ec;
            if (recv(g_conn_sock, &ec, 1, MSG_PEEK | MSG_DONTWAIT) == 0) break;
            usleep(100000);
        }

        if (g_conn_sock >= 0) { close(g_conn_sock); g_conn_sock = -1; }
        pthread_join(t_result, nullptr);
        if (!g_stop) printf("\n[TCP] 서버 재연결 루프로 돌아갑니다...\n");
    }
    printf("[RX] 프로그램을 종료합니다.\n");
    return 0;
}
