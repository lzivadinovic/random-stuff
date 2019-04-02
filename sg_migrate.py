import boto3
from botocore.exceptions import ClientError

profile1='itk'
region1='eu-west-2'
session1 = boto3.Session(profile_name=profile1)
ec2_r1 = session1.client('ec2', region_name = region1)
try:
    response = ec2_r1.describe_security_groups()
except ClientError as e:
    print(e)

def createsg(ec2_session, vpc_id, response)
    try:
        rsp = ec2_session.create_security_group(GroupName=response['GroupName'],
                                             Description=response['Description'],
                                             VpcId=vpc_id)
        security_group_id = rsp['GroupId']
        print('Security Group Created %s in vpc %s.' % (security_group_id, vpc_id))
    
        data = ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            response['IpPermissions'])
        print('Ingress Successfully Set %s' % data)
    except ClientError as e:
        print(e)

region2 = 'eu-west-3'
session2 = boto3.Session(profile_name='neki_profil')
ec2_r2 = session2.client('ec2', region_name = region2)
response2 = ec2_r2.describe_vpcs()
vpc_id = response2.get('Vpcs', [{}])[0].get('VpcId', '')


list(map(createsg, repeat(ec2_r2), repeat(vpc_id), response))
