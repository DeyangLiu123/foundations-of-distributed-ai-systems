/*
 * L68 optional ns-3 experiment: synchronized TCP incast.
 *
 * Validated against the ns-3.47 source API.  Copy this file to scratch/ in an
 * ns-3.47 tree, then run for 8 and 32 senders:
 *
 *   ./ns3 run "scratch/L68_incast --senders=8 --outputPrefix=results/incast-8"
 *   ./ns3 run "scratch/L68_incast --senders=32 --outputPrefix=results/incast-32"
 *
 * Each sender has a 100 Gb/s access link.  All flows converge on one
 * 400 Gb/s bottleneck.  The program writes bottleneck queue occupancy and a
 * per-flow completion-time proxy (application start to final byte received).
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("L68Incast");

namespace
{

struct FlowRecord
{
    uint64_t receivedBytes{0};
    Time completion{Seconds(-1)};
};

std::vector<FlowRecord> g_flows;
std::ofstream g_queueFile;
Time g_applicationStart;

void
QueueBytes(uint32_t oldBytes, uint32_t newBytes)
{
    g_queueFile << Simulator::Now().GetNanoSeconds() << ',' << oldBytes << ',' << newBytes << '\n';
}

void
SinkRx(uint32_t flowId,
       uint64_t targetBytes,
       Ptr<const Packet> packet,
       const Address& /* from */)
{
    auto& record = g_flows.at(flowId);
    record.receivedBytes += packet->GetSize();
    if (record.completion.IsNegative() && record.receivedBytes >= targetBytes)
    {
        record.completion = Simulator::Now();
    }
}

std::string
AccessSubnet(uint32_t index)
{
    // The lesson runs at most 32 senders, so one /24 per sender is sufficient.
    std::ostringstream subnet;
    subnet << "10.1." << (index + 1) << ".0";
    return subnet.str();
}

} // namespace

int
main(int argc, char* argv[])
{
    uint32_t senders = 8;
    uint64_t bytesPerFlow = 10'000'000;
    uint32_t payloadBytes = 1448;
    uint32_t queuePackets = 2000;
    std::string accessRate = "100Gbps";
    std::string bottleneckRate = "400Gbps";
    std::string linkDelay = "1us";
    std::string outputPrefix = "l68-incast";

    CommandLine cmd(__FILE__);
    cmd.AddValue("senders", "Number of synchronized TCP senders", senders);
    cmd.AddValue("bytesPerFlow", "Application bytes sent by each flow", bytesPerFlow);
    cmd.AddValue("payloadBytes", "BulkSend write size in bytes", payloadBytes);
    cmd.AddValue("queuePackets", "Bottleneck DropTail capacity in packets", queuePackets);
    cmd.AddValue("accessRate", "Per-sender access-link rate", accessRate);
    cmd.AddValue("bottleneckRate", "Shared egress-link rate", bottleneckRate);
    cmd.AddValue("linkDelay", "One-way propagation delay per link", linkDelay);
    cmd.AddValue("outputPrefix", "Prefix for queue and FCT CSV files", outputPrefix);
    cmd.Parse(argc, argv);

    if (senders < 2 || senders > 200 || bytesPerFlow == 0 || payloadBytes == 0 ||
        queuePackets == 0)
    {
        std::cerr << "Require 2<=senders<=200 and positive byte/queue arguments" << std::endl;
        return 2;
    }

    NodeContainer sourceNodes;
    sourceNodes.Create(senders);
    NodeContainer routerNode;
    routerNode.Create(1);
    NodeContainer receiverNode;
    receiverNode.Create(1);

    NodeContainer allNodes;
    allNodes.Add(sourceNodes);
    allNodes.Add(routerNode);
    allNodes.Add(receiverNode);
    InternetStackHelper internet;
    internet.Install(allNodes);

    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue(accessRate));
    access.SetChannelAttribute("Delay", StringValue(linkDelay));

    Ipv4AddressHelper address;
    for (uint32_t index = 0; index < senders; ++index)
    {
        NetDeviceContainer devices = access.Install(sourceNodes.Get(index), routerNode.Get(0));
        address.SetBase(Ipv4Address(AccessSubnet(index).c_str()), "255.255.255.0");
        address.Assign(devices);
    }

    PointToPointHelper bottleneck;
    bottleneck.SetDeviceAttribute("DataRate", StringValue(bottleneckRate));
    bottleneck.SetChannelAttribute("Delay", StringValue(linkDelay));
    bottleneck.SetQueue("ns3::DropTailQueue",
                        "MaxSize",
                        QueueSizeValue(QueueSize(std::to_string(queuePackets) + "p")));
    NetDeviceContainer bottleneckDevices =
        bottleneck.Install(routerNode.Get(0), receiverNode.Get(0));
    address.SetBase("10.255.0.0", "255.255.255.0");
    Ipv4InterfaceContainer bottleneckInterfaces = address.Assign(bottleneckDevices);
    Ipv4Address receiverAddress = bottleneckInterfaces.GetAddress(1);

    Ptr<PointToPointNetDevice> routerEgress =
        DynamicCast<PointToPointNetDevice>(bottleneckDevices.Get(0));
    if (!routerEgress || !routerEgress->GetQueue())
    {
        std::cerr << "Unable to locate router egress queue" << std::endl;
        return 3;
    }

    g_queueFile.open(outputPrefix + "-queue.csv", std::ios::out | std::ios::trunc);
    if (!g_queueFile)
    {
        std::cerr << "Unable to open queue output; create the parent directory first" << std::endl;
        return 4;
    }
    g_queueFile << "time_ns,old_bytes,new_bytes\n";
    routerEgress->GetQueue()->TraceConnectWithoutContext("BytesInQueue",
                                                         MakeCallback(&QueueBytes));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    g_flows.assign(senders, FlowRecord{});
    g_applicationStart = Seconds(0.1);
    const Time simulationStop = Seconds(1.0);

    for (uint32_t index = 0; index < senders; ++index)
    {
        const uint16_t port = static_cast<uint16_t>(9000 + index);
        PacketSinkHelper sink("ns3::TcpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApplication = sink.Install(receiverNode.Get(0));
        sinkApplication.Start(Seconds(0.0));
        sinkApplication.Stop(simulationStop);

        Ptr<PacketSink> packetSink = DynamicCast<PacketSink>(sinkApplication.Get(0));
        packetSink->TraceConnectWithoutContext(
            "Rx", MakeBoundCallback(&SinkRx, index, bytesPerFlow));

        BulkSendHelper source("ns3::TcpSocketFactory",
                              InetSocketAddress(receiverAddress, port));
        source.SetAttribute("MaxBytes", UintegerValue(bytesPerFlow));
        source.SetAttribute("SendSize", UintegerValue(payloadBytes));
        ApplicationContainer sourceApplication = source.Install(sourceNodes.Get(index));
        sourceApplication.Start(g_applicationStart);
        sourceApplication.Stop(simulationStop);
    }

    Simulator::Stop(simulationStop);
    Simulator::Run();
    Simulator::Destroy();
    g_queueFile.close();

    std::ofstream fctFile(outputPrefix + "-fct.csv", std::ios::out | std::ios::trunc);
    if (!fctFile)
    {
        std::cerr << "Unable to open FCT output" << std::endl;
        return 5;
    }
    fctFile << "flow_id,received_bytes,completed,fct_ms\n";
    std::vector<double> completedFctMs;
    for (uint32_t index = 0; index < senders; ++index)
    {
        const auto& record = g_flows.at(index);
        const bool completed = !record.completion.IsNegative();
        const double fctMs = completed ? (record.completion - g_applicationStart).GetSeconds() * 1000
                                       : -1.0;
        fctFile << index << ',' << record.receivedBytes << ',' << (completed ? 1 : 0) << ','
                << std::fixed << std::setprecision(6) << fctMs << '\n';
        if (completed)
        {
            completedFctMs.push_back(fctMs);
        }
    }
    fctFile.close();

    if (completedFctMs.empty())
    {
        std::cerr << "No flow completed before simulationStop" << std::endl;
        return 6;
    }
    const double mean = std::accumulate(completedFctMs.begin(), completedFctMs.end(), 0.0) /
                        completedFctMs.size();
    const double maximum = *std::max_element(completedFctMs.begin(), completedFctMs.end());
    std::cout << std::fixed << std::setprecision(3) << "senders=" << senders
              << " completed=" << completedFctMs.size() << '/' << senders
              << " mean_fct_ms=" << mean << " max_fct_ms=" << maximum << '\n'
              << "queue_csv=" << outputPrefix << "-queue.csv\n"
              << "fct_csv=" << outputPrefix << "-fct.csv" << std::endl;
    return completedFctMs.size() == senders ? 0 : 7;
}
